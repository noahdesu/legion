#include "legion_io.h"
#include "hdf5.h"

void copy_values_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, HighLevelRuntime *runtime)
{
  bool write = *(bool*)task->args;
  Piece piece = *(Piece*)task->local_args;
  
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(piece.child_lr == regions[0].get_logical_region()); 

  // not needed because task is launched with no instance flag set...? doesn't
  // seem to work without doing this.
  runtime->unmap_region(ctx, regions[0]);

  std::map<FieldID, const char*> field_map;
  field_map.insert(std::make_pair(FID_TEMP, "bam/baz"));

  PhysicalRegion pr = runtime->attach_rados(ctx, piece.shard_name,
      piece.child_lr, piece.child_lr,
      field_map, write ? LEGION_FILE_READ_WRITE: LEGION_FILE_READ_ONLY);

  runtime->remap_region(ctx, pr);
  pr.wait_until_valid();
  
  LogicalRegion src, dst;
  if (write) { 
    src = regions[1].get_logical_region();
    dst = piece.child_lr;
  } else {
    dst = regions[1].get_logical_region();
    src = piece.child_lr;
  } 

  CopyLauncher copy_launcher;
  copy_launcher.add_copy_requirements(
      RegionRequirement(src, READ_ONLY, EXCLUSIVE, src),
      RegionRequirement(dst, WRITE_DISCARD, EXCLUSIVE, dst));

  copy_launcher.add_src_field(0, FID_TEMP);
  copy_launcher.add_dst_field(0, FID_TEMP);

  runtime->issue_copy_operation(ctx, copy_launcher);
  
  runtime->detach_rados(ctx, pr);
}

void PersistentRegion::write_persistent_subregions(Context ctx,
    LogicalRegion src_lr, LogicalPartition src_lp)
{
  ArgumentMap arg_map;
  for (std::vector<Piece>::const_iterator it = pieces.begin(); 
      it != pieces.end(); it++) {
    Piece p = *it;
    arg_map.set_point(p.dp, TaskArgument(&p, sizeof(p)));
  }

  bool copy_write = true;
  IndexLauncher write_launcher(COPY_VALUES_TASK_ID, this->dom,
      TaskArgument(&copy_write, sizeof(bool)), arg_map);
  
  write_launcher.add_region_requirement(RegionRequirement(
        this->lp, 0, READ_WRITE, EXCLUSIVE, this->parent_lr));

  write_launcher.add_region_requirement(RegionRequirement(
        src_lp, 0, READ_WRITE, EXCLUSIVE, src_lr));
  
  for (std::map<FieldID, const char*>::const_iterator it = field_map.begin();
       it != field_map.end(); it++) {
    FieldID fid = it->first;
    write_launcher.region_requirements[0].add_field(fid, false /* no instance required */);
    write_launcher.region_requirements[1].add_field(fid);
  }

  runtime->execute_index_space(ctx, write_launcher); 
}

void PersistentRegion::read_persistent_subregions(Context ctx,
    LogicalRegion src_lr, LogicalPartition src_lp)
{
  ArgumentMap arg_map;
  for (std::vector<Piece>::const_iterator it = pieces.begin(); 
      it != pieces.end(); it++) {
    Piece p = *it;
    arg_map.set_point(p.dp, TaskArgument(&p, sizeof(p)));
  }

  bool copy_write = false;
  IndexLauncher read_launcher(COPY_VALUES_TASK_ID, this->dom,
      TaskArgument(&copy_write, sizeof(bool)), arg_map);
  
  read_launcher.add_region_requirement(RegionRequirement(
        this->lp, 0, READ_WRITE, EXCLUSIVE, this->parent_lr));
  
  read_launcher.add_region_requirement(RegionRequirement(
        src_lp, 0, READ_WRITE, EXCLUSIVE, src_lr));
  
  for (std::map<FieldID, const char*>::const_iterator it = field_map.begin();
       it != field_map.end(); it++) {
    FieldID fid = it->first;
    read_launcher.region_requirements[0].add_field(fid, false /* no instance required */);
    read_launcher.region_requirements[1].add_field(fid);
  }

  runtime->execute_index_space(ctx, read_launcher); 
}

/*
 *
 */
void PersistentRegion::create_persistent_subregions(Context ctx,
    const char *name, LogicalRegion parent_lr, LogicalPartition lp,
    Domain dom, std::map<FieldID, const char*> &field_map)
{
  this->lp = lp;
  this->parent_lr = parent_lr;
  this->field_map = field_map;
  this->dom = dom;

  for (LegionRuntime::LowLevel::Domain::DomainPointIterator itr(dom); itr; itr++) {

    LegionRuntime::LowLevel::DomainPoint dp = itr.p;

    Piece piece;
    piece.dp = dp;
    piece.parent_lr = parent_lr;
    piece.child_lr = runtime->get_logical_subregion_by_color(ctx, lp, dp);

    IndexSpace is = runtime->get_index_subspace(ctx,
        lp.get_index_partition(), dp); 
    FieldSpace fs = piece.child_lr.get_field_space();
    Domain d = runtime->get_index_space_domain(ctx, is);
    int dim = d.get_dim();

    int x_min = 0, y_min = 0, z_min = 0,
        x_max = 0, y_max = 0, z_max = 0;

    int shard_size[dim];
    int shard_dims[dim*2];

    LegionRuntime::HighLevel::RadosMetadata obj_md;
    obj_md.ndim = dim;

    std::ostringstream ds_name_stream;

    switch(dim) {
      case 2:
        {
          ds_name_stream <<  piece.dp[0] << "-" << piece.dp[1];
          sprintf(piece.shard_name, "%d-%d-%s",
              piece.dp[0], piece.dp[1], name); 

          x_min = d.get_rect<2>().lo.x[0];
          y_min = d.get_rect<2>().lo.x[1];
          x_max = d.get_rect<2>().hi.x[0];
          y_max = d.get_rect<2>().hi.x[1];

          // shape of the shard
          shard_size[0] = x_max-x_min+1;
          shard_size[1] = y_max-y_min+1;

          obj_md.size[0] = shard_size[0];
          obj_md.size[1] = shard_size[1];

          // global coordinate of shard
          shard_dims[0] = x_min;
          shard_dims[1] = y_min;
          shard_dims[2] = x_max;
          shard_dims[3] = y_max;

          obj_md.min_bounds[0] = shard_dims[0];
          obj_md.min_bounds[1] = shard_dims[1];
          obj_md.max_bounds[0] = shard_dims[2];
          obj_md.max_bounds[1] = shard_dims[3];
        }
        break;

      case 3:
        {
          ds_name_stream <<  piece.dp[0] << "-" << piece.dp[1] << "-" << piece.dp[2];
          sprintf(piece.shard_name, "%d-%d-%d-%s",
              piece.dp[0], piece.dp[1], piece.dp[2], name); 

          x_min = d.get_rect<3>().lo.x[0];
          y_min = d.get_rect<3>().lo.x[1];
          z_min = d.get_rect<3>().lo.x[2];
          x_max = d.get_rect<3>().hi.x[0];
          y_max = d.get_rect<3>().hi.x[1];
          z_max = d.get_rect<3>().hi.x[2];

          // shape of the shard
          shard_size[0] = x_max-x_min+1;
          shard_size[1] = y_max-y_min+1;
          shard_size[2] = z_max-z_min+1;

          obj_md.size[0] = shard_size[0];
          obj_md.size[1] = shard_size[1];
          obj_md.size[2] = shard_size[2];

          // global coordinate of shard
          shard_dims[0] = x_min;
          shard_dims[1] = y_min;
          shard_dims[2] = z_min;
          shard_dims[3] = x_max;
          shard_dims[4] = y_max;
          shard_dims[5] = z_max;

          obj_md.min_bounds[0] = shard_dims[0];
          obj_md.min_bounds[1] = shard_dims[1];
          obj_md.min_bounds[2] = shard_dims[2];
          obj_md.max_bounds[0] = shard_dims[3];
          obj_md.max_bounds[1] = shard_dims[4];
          obj_md.max_bounds[2] = shard_dims[5];
        }
        break;

      default:
        assert(false);
    }

    pieces.push_back(piece);

    assert(field_map.size() == 1);

    std::map<FieldID, const char*>::const_iterator it = field_map.begin();
    assert(it != field_map.end());

    const char *path = it->second;
    size_t field_size = runtime->get_field_size(ctx, fs, it->first);
    obj_md.field_size = field_size;

    std::stringstream field_filename;
    field_filename << piece.shard_name << "." << path;
    ioctx->remove(field_filename.str());
    int ret = ioctx->create(field_filename.str(), true);
    assert(ret == 0);

    ceph::bufferlist bl;
    bl.append((char*)&obj_md, sizeof(obj_md));
    ret = ioctx->omap_set_header(field_filename.str(), bl);
    assert(ret == 0);
  }
}

void PersistentRegion_init() {
  HighLevelRuntime::register_legion_task<copy_values_task>(COPY_VALUES_TASK_ID,
      Processor::LOC_PROC, true /*single*/, true /*index*/);
}

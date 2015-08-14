#include "legion_io.h"
#include "hdf5.h"

void split_path_file(char** p, char** f, const char *pf);


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
  hid_t link_file_id, shard_group_id, shard_ds_id, dataspace_id, dtype_id, shard_file_id, attr_ds_id, link_group_id, link_group_2_id;
  herr_t status;
  link_file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  dtype_id  = H5Tcopy (H5T_NATIVE_DOUBLE);

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
    sprintf(piece.shard_name, "%d-%d-%s", piece.dp[0], piece.dp[1], name); 
    pieces.push_back(piece);

    FieldSpace fs = piece.child_lr.get_field_space();

    int x_min = 0, y_min = 0, z_min = 0,
        x_max = 0, y_max = 0, z_max = 0;

    shard_file_id = H5Fcreate(piece.shard_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    IndexSpace is = runtime->get_index_subspace(ctx,
        lp.get_index_partition(), dp); 
    Domain d = runtime->get_index_space_domain(ctx, is);
    int dim = d.get_dim();

    int *shard_dims;

    switch(dim) {
      case 2:
        {
          x_min = d.get_rect<2>().lo.x[0];
          y_min = d.get_rect<2>().lo.x[1];
          x_max = d.get_rect<2>().hi.x[0];
          y_max = d.get_rect<2>().hi.x[1];

          hsize_t dims[2];
          dims[0] = x_max-x_min+1;
          dims[1] = y_max-y_min+1;
          dataspace_id = H5Screate_simple(2, dims, NULL);

          dims[0] = 2; dims[1] = 2; 
          attr_ds_id = H5Screate_simple(2, dims, NULL);
          shard_dims = (int*) malloc(4*sizeof(int)); 
          shard_dims[0] = x_min;
          shard_dims[1] = y_min;
          shard_dims[2] = x_max;
          shard_dims[3] = y_max;
        }
        break;

      default:
        assert(false);
    }

    for (std::map<FieldID, const char*>::const_iterator it = field_map.begin();
         it != field_map.end(); it++) {

      const char *path = it->second;
      size_t field_size = runtime->get_field_size(ctx, fs, it->first);

      std::stringstream field_filename;
      field_filename << piece.shard_name << "." << path;
      int ret = ioctx->create(field_filename.str(), false);
      assert(ret == 0);




      char* ds;
      char* gp;
      split_path_file(&gp, &ds, path);

      status = H5Tset_size(dtype_id, field_size);
      if(H5Lexists(shard_file_id, gp, H5P_DEFAULT)) { 
        shard_group_id = H5Gopen2(shard_file_id, gp, H5P_DEFAULT);
      } else { 
        shard_group_id = H5Gcreate2(shard_file_id, gp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      } 
      if(H5Lexists(shard_group_id, ds, H5P_DEFAULT)) { 
        shard_ds_id = H5Dopen2(shard_group_id, ds, H5P_DEFAULT); 
      } else { 
        shard_ds_id = H5Dcreate2(shard_group_id, ds, H5T_NATIVE_DOUBLE,
            dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT);
      }

      if(H5Lexists(link_file_id, gp, H5P_DEFAULT) && H5Lexists(link_file_id, it->second, H5P_DEFAULT)) {
        link_group_2_id = H5Gopen2(link_file_id, it->second, H5P_DEFAULT);
      } else {
        link_group_id = H5Gcreate2(link_file_id, gp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        link_group_2_id = H5Gcreate2(link_group_id, ds, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(link_group_id);
      }

      std::ostringstream ds_name_stream;

      ds_name_stream <<  piece.dp[0] << "-" << piece.dp[1];

      hid_t attr_id = H5Acreate2(shard_ds_id, "dims", H5T_NATIVE_INT, attr_ds_id,
          H5P_DEFAULT, H5P_DEFAULT);

      status = H5Awrite(attr_id, H5T_NATIVE_INT, shard_dims);
      H5Aclose(attr_id);

      H5Dclose(shard_ds_id);
      H5Gclose(shard_group_id);
      H5Fclose(shard_file_id);


      status = H5Lcreate_external(piece.shard_name, it->second,
          link_group_2_id, ds_name_stream.str().c_str(),
          H5P_DEFAULT, H5P_DEFAULT);

      shard_file_id = H5Fopen(piece.shard_name, H5F_ACC_RDWR, H5P_DEFAULT);      
      H5Gclose(link_group_2_id);
    }
    H5Fclose(shard_file_id);

  }
  H5Fclose(link_file_id);
}


void split_path_file(char** p, char** f, const char *pf) {
    char *slash = (char*)pf, *next;
    while ((next = strpbrk(slash + 1, "\\/"))) slash = next;
    if (pf != slash) slash++;
    *p = strndup(pf, slash - pf);
    *f = strdup(slash);
}

void PersistentRegion_init() {
  HighLevelRuntime::register_legion_task<copy_values_task>(COPY_VALUES_TASK_ID,
      Processor::LOC_PROC, true /*single*/, true /*index*/);
}
        

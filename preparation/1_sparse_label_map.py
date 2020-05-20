# %%
import pandas
import pickle

# %%
def create_label_map(scannet_path, ndsis_path, label_map_path, name_map_path):
    if False:
        df_scannet = pandas.read_csv(scannet_path, sep='\t')
        df_3dsis = pandas.read_csv(ndsis_path)
        df_3dsis['mappedIdConsecutive'] -= 1
        df_3dsis.loc[df_3dsis.nyu40id.isin({1, 2, 22}), 'mappedIdConsecutive'] = -1
        df_3dsis.loc[
            df_3dsis.mappedIdConsecutive == 18, 'nyu40class'] = 'otherprop'
        df_scannet_reduced = df_scannet[['raw_category', 'nyu40id', 'nyu40class']]
        df_3dsis_reduced = df_3dsis[
            ['nyu40id', 'mappedIdConsecutive', 'weight']].copy()

        df_merged = df_scannet_reduced.merge(
            df_3dsis_reduced, on='nyu40id', how='outer').drop(columns='nyu40id')
        df_raw_map = df_merged.set_index('raw_category')
        df_name_map = df_raw_map.reset_index(drop=True).drop_duplicates()
        df_name_map = df_name_map.rename(
            columns={'nyu40class': 'name', 'mappedIdConsecutive': 'class_id'})
        df_name_map = df_name_map.set_index('class_id').sort_index()
        df_raw_map = df_raw_map.drop(columns=['nyu40class', 'weight'])
        df_raw_map = df_raw_map.rename(columns={'mappedIdConsecutive': 'class_id'})
    else:
        df_scannet = pandas.read_csv(scannet_path, sep='\t')
        df_name_map = df_scannet[['nyu40id', 'nyu40class']].drop_duplicates()
        df_name_map = df_name_map.rename(
            columns={'nyu40class': 'name', 'nyu40id': 'class_id'})
        df_name_map = df_name_map.set_index('class_id').sort_index()
        df_raw_map = df_scannet[['raw_category', 'nyu40id']]
        df_raw_map = df_raw_map.rename(columns={'nyu40id': 'class_id'})
        df_raw_map = df_raw_map.set_index('raw_category')

    series_raw_map = df_raw_map['class_id']
    df_name_map.to_csv(name_map_path)

    print(series_raw_map.head)
    print(df_name_map['name'].values)

    with open(label_map_path, 'wb') as handle:
        pickle.dump(
            series_raw_map.to_dict(), handle,
            protocol=pickle.HIGHEST_PROTOCOL)
# %%

if __name__ == '__main__':
    from scannet_config.pathes import( 
        name_map_path, label_map_path, scannet_path, ndsis_path)

    create_label_map(scannet_path, ndsis_path, label_map_path, name_map_path)


# %%

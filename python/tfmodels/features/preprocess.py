def gen_stats_dicts(tbl, model_dir, frequency_limit=25):
    stats = {}
    for c in embed_cols + cat_cols + num_cols:
        min = tbl.select(c).get_stats(c, "min")[c]
        max = tbl.select(c).get_stats(c, "max")[c]
        total_num = tbl.select(c).size()
        dist_num = tbl.select(c).distinct().size()
        null_num = tbl.filter(col(c).isNull()).size()
        stats[c + "_min"] = min
        stats[c + "_max"] = max
        stats[c + "_total_num"] = total_num
        stats[c + "_ditinct_num"] = dist_num
        stats[c + "_null_num"] = null_num
        stats[c + "_null_ratio"] = null_num / float(total_num)

    for c in ts_cols + ["label"]:
        label0_cnt = tbl.select(c).filter(col(c) == 0).size()
        label1_cnt = tbl.select(c).filter(col(c) > 0).size()
        stats[c + "_label0_cnt"] = label0_cnt
        stats[c + "_label1_cnt"] = label1_cnt

    # for item in stats.items():
    #     print(item)
    stats_dir = model_dir + "/stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    with open(os.path.join(stats_dir, "feature_stats"), 'wb') as f:
        pickle.dump(stats, f)

    index_dicts=[]
    for c in embed_cols:
        c_count = tbl.select(c).group_by(c, agg={c: "count"}).rename({"count(" + c + ")": "count"})
        c_count.df_scaled.printSchema()
        c_count = c_count.filter(col("count") >= frequency_limit).order_by("count", ascending=False)
        c_count_pd = c_count.to_pandas()
        c_count_pd.reindex()
        c_count_pd[c + "_new"] = c_count_pd.index + 1
        index_dict = dict(zip(c_count_pd[c], c_count_pd[c + "_new"]))
        index_dict_reverse = dict(zip(c_count_pd[c + "_new"], c_count_pd[c]))
        index_dicts.append(index_dict)
        with open(os.path.join(stats_dir, c + "_index_dict.txt"), 'w') as text_file:
            c_count_pd.to_csv(text_file, index=False)
        with open(os.path.join(stats_dir, c + "_index_dict"), 'wb') as f:
            pickle.dump(index_dict, f)
        with open(os.path.join(stats_dir, c + "_index_dict_reverse"), 'wb') as f:
            pickle.dump(index_dict_reverse, f)
    return stats, index_dicts
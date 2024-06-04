import factorlab as lab


name = "volume_weighted_price"
start = "20230104"
stop = "20230104"
factor_data = lab.factor.get(name, start=start, stop=stop, n_jobs=1)

print(lab.factor.perform_crosssection(factor_data, date="20230104", period=5, image="out/cross_section.png"))
print(lab.factor.perform_inforcoef(factor_data, period=5, start=start, stop=stop, image="out/infor_coef.png"))
print(lab.factor.perform_grouping(factor_data, period=5, start=start, stop=stop, image="out/grouping.png"))
print(lab.factor.perform_topk(factor_data, period=5, start=start, stop=stop, image="out/topk.png"))

if factor_data.index.nlevels == 1:
    factor_data = factor_data.stack()
    factor_data.index.names = [lab.factor._code_level, lab.factor._date_level]
factor_data = factor_data.reorder_levels([lab.factor._code_level, lab.factor._date_level])

if name not in lab.factor.columns:
    lab.factor.add({name: factor_data.dtype})
lab.factor.update(factor_data)

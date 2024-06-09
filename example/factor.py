import factorlab as lab


name = "head_weighted_price"
start = "20100101"
stop = "20240604"
factor = lab.Factor("./data/prices", create=True, code_level="code", date_level="date")

factor_data = factor.get(name, start=start, stop=stop, n_jobs=-1)

print(lab.factor.perform_crosssection(factor_data, date="20230104", period=5, image="out/cross_section.png"))
print(lab.factor.perform_inforcoef(factor_data, period=5, start=start, stop=stop, image="out/infor_coef.png"))
print(lab.factor.perform_grouping(factor_data, period=5, start=start, stop=stop, image="out/grouping.png"))
print(lab.factor.perform_topk(factor_data, period=5, start=start, stop=stop, image="out/topk.png"))

if factor_data.index.nlevels == 1:
    factor_data = factor_data.stack()
    factor_data.index.names = [factor._date_level, factor._code_level]
factor_data = factor_data.reorder_levels([factor._code_level, factor._date_level])
factor_data.name = name

if name not in lab.factor.columns:
    factor.add({name: factor_data.dtype})
factor.update(factor_data)

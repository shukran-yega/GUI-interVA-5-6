# -*- coding: utf-8 -*-

from vacheck.datacheck5 import datacheck5, get_probbase
from rpy2.robjects.packages import data, importr
import rpy2.robjects as robjects
from rpy2.robjects.conversion import get_conversion, localconverter
from rpy2.robjects import pandas2ri
from numpy import nan

r_iva5 = importr("InterVA5")
randomva5 = data(r_iva5).fetch("RandomVA5")["RandomVA5"]
probbasev5 = data(r_iva5).fetch("probbaseV5")["probbaseV5"]

robjects.r("data(probbaseV5, package='InterVA5')")
robjects.r("data(RandomVA5, package='InterVA5')")
robjects.r('''
  pb5 <- as.matrix(probbaseV5)
  ra5 <- as.matrix(RandomVA5)
  ra5[ra5 == "n"] <- "0"
  ra5[ra5 == "y"] <- "1"
  ra5[!(ra5 %in% c("1", "0"))] <- NA
  output <- matrix(NA, nrow=nrow(ra5), ncol=ncol(ra5))
  for (i in 1:nrow(ra5)) {
    input <- as.numeric(ra5[i,])
    output[i,] <- InterVA5::DataCheck5(Input=input,
                                       id=RandomVA5[i, 1],
                                       probbaseV5=pb5,
                                       write=FALSE)$Output
  }
  r_df <- as.data.frame(output, colnames = colnames(ra5))
''')
r_df = robjects.globalenv["r_df"]

with localconverter(robjects.default_converter + pandas2ri.converter):
    r_check = get_conversion().rpy2py(r_df)
    r_data = get_conversion().rpy2py(randomva5)

for col in r_data:
    if not (col == "ID"):
        r_data[col] = r_data[col].cat.rename_categories({"y": 1, "n": 0, ".": 3})
        r_data[col] = r_data[col].astype("float64")
        r_data[col] = r_data[col].replace(3, nan)


pb = get_probbase()
py_check = r_data.replace(".", nan).apply(
    lambda x: datacheck5(x, x.ID, probbase=pb)['output'],
    axis=1)
py_check
r_check = r_check.set_axis(list(py_check), axis=1)
r_check["ID"] = py_check["ID"].copy()


def test_r_comparison():
    assert(all(r_check.replace(nan, 2).eq(py_check.replace(nan, 2))))

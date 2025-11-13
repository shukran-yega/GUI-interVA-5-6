# -*- coding: utf-8 -*-

import pytest
from pandas import read_csv, Series
from numpy import nan
import pkgutil
from io import BytesIO

from vacheck.datacheck5 import datacheck5, get_probbase
from vacheck import exceptions

va_data_csv = pkgutil.get_data("vacheck", "data/example_input.csv")
va_data = read_csv(BytesIO(va_data_csv))
pb = get_probbase()


@pytest.fixture
def single_record():
    return va_data.iloc[0].copy()


class TestReturnDefaults:

    va_id = va_data["ID"][0]
    results = datacheck5(va_input=va_data.iloc[0],
                         probbase=pb,
                         va_id=va_id)
    output = results["output"]
    output_2 = output.replace(nan, 2)

    def test_return_types(self):
        assert type(self.results) is dict
        assert type(self.results["output"]) == Series
        assert type(self.results["first_pass"]) == list
        assert type(self.results["second_pass"]) == list

    def test_output_indices(self):
        assert self.output.index.equals(va_data.columns)

    def test_output_values(self):
        assert all(self.output_2.isin([self.va_id, 0, 1, 2]))


def test_invalid_arg_type(single_record):
    with pytest.raises(exceptions.VAInputException):
        datacheck5(single_record.to_list(), probbase=pb, va_id="d1")
    with pytest.raises(exceptions.VAIDException):
        single_record.iloc[0] = ""
        datacheck5(single_record, probbase=pb, va_id="")
    with pytest.raises(exceptions.VAInputException):
        single_record.iloc[0] = 3
        single_record.iloc[2] = 3
        datacheck5(single_record, probbase=pb, va_id=3)


def test_invalid_va_input_data_value(single_record):
    bad_record = single_record
    bad_record.iloc[2] = 33
    with pytest.raises(exceptions.VAInputException):
        datacheck5(bad_record, probbase=pb, va_id="d1")


def test_invalid_va_input_n_elements(single_record):
    bad_record = single_record
    bad_record.pop("i004a")
    with pytest.raises(exceptions.VAInputException):
        datacheck5(bad_record, probbase=pb, va_id="d1")

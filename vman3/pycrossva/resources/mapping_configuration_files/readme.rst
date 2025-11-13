Configuration Files
====================

CrossVA runs by applying the specified mappings in its configuration files to
the raw data provided. The package comes with some default configurations
which map common inputs to common outputs, but it is possible to create your
own customized version.

These files must have the following columns:

  * New Column Name
  * New Column Documentation
  * Source Column ID
  * Source Column Documentation
  * Relationship
  * Condition
  * Prerequisite

Of those, only `New Column Name`, `Source Column ID`, `Relationship`, and
`Condition` must be filled out in every row.

Each row in the configuration mapping gives instructions to map information from
a single column in the raw input data to a new column in the final transformed
data.

Each row can be read as roughly:
The new column `New Column Name` will get the value True in rows where the column
`Source Column ID` in the input data is `Relationship` to `Condition`,
and, optionally, the `Prerequisite` column in the output data is also True.

When the input data is NA, the NAs will be preserved through the transformation.
That is, no matter the `Relationship` and `Condition`, if the value in the source
column is NA, then the result will be NA, instead of a boolean.

.. note::
  If the same column name has multiple conditions specified, then each operation
  will only update the pre-existing column. False and NA values in the
  transformed data will be overwritten if a prior condition was false, but a new
  condition was true.

  This creates an implicit `OR` relationship between the different conditions
  listed.


.. topic:: Structure of Mapping Configuration table

  +------------------+--------------------------------------------------------------+------------------+-------------------------------------+--------------+---------------------------------------+--------------+
  |  New Column Name | New Column Documentation                                     | Source Column ID | Source Column Documentation         | Relationship | Condition                             | Prerequisite |
  +==================+==============================================================+==================+=====================================+==============+=======================================+==============+
  | DEL_ELSE         | Did she give birth elsewhere, e.g. on the way to a facility? | Id10337          | (Id10337) Where did she give birth? | eq           | other                                 |              |
  +------------------+--------------------------------------------------------------+------------------+-------------------------------------+--------------+---------------------------------------+--------------+
  | DEL_ELSE         | Did she give birth elsewhere, e.g. on the way to a facility? | Id10337          | (Id10337) Where did she give birth? | eq           | on_route_to_hospital_or_facility      |              |
  +------------------+--------------------------------------------------------------+------------------+-------------------------------------+--------------+---------------------------------------+--------------+

  * The first row indicates that where the column 'Id10337' is equal to other in the input data,
    then 'DEL_ELSE' is true in the output data.

  * The second row indicates that where the column 'Id10337' is equal to
    'on_route_to_hospital_or_facility' in the input data,
    then DEL_ELSE is true in the output data.

  In this sense, if 'Id10337' is 'on_route_to_hospital_or_facility', 'DEL_ELSE' will be set
  to False on the first condition, and then updated to True on the second condition.
  However, if 'Id10337' is `other`, then 'DEL_ELSE' will be set to True on the first
  condition, and remain True (not updated) on the second condition, regardless of
  the value in the input data, since a condition for the new column has already
  been satisfied.


.. topic:: New Column Name

  The `New Column Name` column should contain the name of the
  new column to be created in the final data. All of the columns required by the
  intended algorithm should be listed, with corresponding documentation in the
  New Column Documentation column if possible.

.. topic:: New Column Documentation

  The `New Column Documentation` column should contain a brief statement
  explaining what the new column is meant to represent.

.. topic:: Source Column ID

  The `Source Column ID` column should contain the unique
  identifier at the end of the column name in the input data. It should only be
  left blank in cases where the New Column being created (and required by the intended algorithm)
  depends on information that is unavailable in the source data, and thus there is
  no relevant source column.

.. topic:: Source Column Documentation

  The `Source Column Documentation` column should contain a
  brief statement explaining what information the source column contains. This,
  along with the `New Column Documentation` column, makes it much easier to
  at-a-glance check the logic behind these mappings.

.. topic:: Relationship

  The `Relationship` column should contain one of 8 valid
  relationships, which use the value in the `Condition` column to return a
  boolean value for the output data. The currently supported relationships are:

  * eq: is equal to
  * gt: is greater than
  * ge: is greater than or equal to
  * lt: is less than
  * le: is less than or equal to
  * ne: is not equal to
  * contains: contains the substring
  * between: is between 2 numbers, inclusive

.. topic:: Condition

  The `Condition` column should contain the condition being applied to the
  source column. For example, `yes`, `5` or `15 to 30`.

.. note::

  Conditions in the form ## to ## should only be used when the relationship is
  between, in order to give the two numbers that make up the low and high end
  of the acceptable range.

.. topic:: Prerequisite

  The `Prerequisite` column is optional. It should be left blank if there is
  no prerequisite. If there is a prerequisite condition, then this column should
  contain the name of the column in the final data to reference.

  For example, the new column `MAGEGP1` is created based on the condition of if
  the source column `ageInYears` is `between` `12 to 19`. It also lists a prerequisite
  of `FEMALE`, which is a previously created column in the output data,
  containing its own boolean, which checks to see if Id10019 is equal to "female".

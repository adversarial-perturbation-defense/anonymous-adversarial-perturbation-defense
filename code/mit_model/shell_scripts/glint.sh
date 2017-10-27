#!/bin/bash
echo 'Pylint with google style guide...'
python $(which pylint) -E --rcfile=./googlecl-pylint.rc $1

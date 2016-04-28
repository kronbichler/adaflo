To create new tests in adaflo, consider the following hints:

1. You add a new source file named "test_me.cc"
2. You add one or more parameter files that test various aspects while running the code compiled by "test_me.cc". The parameter files need to start with the same string and can then extend over that. The extension must be .prm. These parameter files determine the actual name of the test, TEST_NAME. If you want to associate two parameter files, you can name them e.g. "test_me_1.prm" and "test_me_2.prm".
3. To each parameter file, add a text file with the same name as the parameter file and .output extension, e.g. "test_me_1.output" and "test_me_2.output", that show the expected output from the test. We currently only test for the output we obtain.


When making changes to the code, make sure to run the tests to see whether you change functionality. If you change functionality and have verified that the new output is 'more correct' or 'more appropriate', copy the output of the test, typically found in output-TEST_NAME/screen-output after running the test, to the file TEST_NAME.output

When adding new files, these might not always be detected by cmake. Run "cmake" in the adaflo base directory to re-create a list of tests.

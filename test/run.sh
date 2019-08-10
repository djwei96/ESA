rm -r summ
mkdir summ
cp -r ../model/dbpedia/ output
cp -r ../model/lmdb/ output
java -jar esummeval_v1.1.jar ESBM_benchmark_v1.1/ output/


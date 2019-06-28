rm -r summ
mkdir summ
cp -r ../model/dbpedia/ summ
cp -r ../model/lmdb/ summ
java -jar esummeval_v1.1.jar ESBM_benchmark_v1.1/ summ/


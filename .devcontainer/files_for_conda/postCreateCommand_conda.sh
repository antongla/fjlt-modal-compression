PARENTDIR="${PWD}"
chmod +x ${PARENTDIR}/build_missing_directories.sh
bash ${PARENTDIR}/build_missing_directories.sh

conda install -y numpy scipy matplotlib jupyter tqdm h5py
pip install pydmd

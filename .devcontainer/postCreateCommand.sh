PARENTDIR="${PWD}"
chmod +x ${PARENTDIR}/build_missing_directories.sh
bash ${PARENTDIR}/build_missing_directories.sh

sudo apt-get update && sudo apt-get install -y libhdf5-dev
pip install -r requirements.txt

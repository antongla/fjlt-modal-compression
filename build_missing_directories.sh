PARENTDIR="${PWD}/notebooks"
message="Bulding missing untracked data and figure directories in ${PARENTDIR}"
echo $message
mkdir --parents "${PARENTDIR}/data" "${PARENTDIR}/svd/data" "${PARENTDIR}/performance-tests/data" "${PARENTDIR}/dmd/data"
mkdir --parents "${PARENTDIR}/svd/figures" "${PARENTDIR}/dmd/figures" "${PARENTDIR}/performance-tests/figures"

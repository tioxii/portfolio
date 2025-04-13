# Baseline
set(PROGRAM_NAME reduction)
set(SOURCES reduction-nocluster.cu)
add_executable(${PROGRAM_NAME} ${SOURCES})

# Reduction with max clusters
set(PROGRAM_NAME reduction-cluster)
set(SOURCES reduction-cluster.cu)
add_executable(${PROGRAM_NAME} ${SOURCES})
set_target_properties(${PROGRAM_NAME} PROPERTIES CUDA_ARCHITECTURES "90")
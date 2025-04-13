# Baseline
set(PROGRAM_NAME smem)
set(SOURCES smem.cu)
add_executable(${PROGRAM_NAME} ${SOURCES})

# kmeans with clusters
# set(PROGRAM_NAME kmeans-TBC)
# set(SOURCES kmeans-TBC.cu)
# add_executable(${PROGRAM_NAME} ${SOURCES})
# set_target_properties(${PROGRAM_NAME} PROPERTIES CUDA_ARCHITECTURES "90")


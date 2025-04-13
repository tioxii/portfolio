# Baseline
set(PROGRAM_NAME kmeans)
set(SOURCES kmeans.cu)
add_executable(${PROGRAM_NAME} ${SOURCES})
target_compile_options(${PROGRAM_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)


# kmeans with clusters
#set(PROGRAM_NAME kmeans-TBC)
#set(SOURCES kmeans-TBC.cu)
#add_executable(${PROGRAM_NAME} ${SOURCES})
#set_target_properties(${PROGRAM_NAME} PROPERTIES CUDA_ARCHITECTURES "90")

# kmeans with clusters as limit for k
set(PROGRAM_NAME kmeans-TBC-limited)
set(SOURCES kmeans-TBC-limited.cu)
add_executable(${PROGRAM_NAME} ${SOURCES})
set_target_properties(${PROGRAM_NAME} PROPERTIES CUDA_ARCHITECTURES "90")

# Baseline long
set(PROGRAM_NAME kmeans-ulong)
set(SOURCES kmeans-ulong.cu)
add_executable(${PROGRAM_NAME} ${SOURCES})
set(PROGRAM_NAME Battleships)
set(SOURCES main.c saveGame.c SDLship.c shipAI.c shipFunc.c shipOptions.c)
add_executable(${PROGRAM_NAME} ${SOURCES})

target_link_libraries(${PROGRAM_NAME}
        PRIVATE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

add_custom_command(TARGET Battleships PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/options.txt $<TARGET_FILE_DIR:Battleships>)
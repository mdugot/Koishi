default: all

############################################################################
##############################    FOR C++    ###############################
############################################################################

NAME= testKoishi
CC= g++
CFLAGS= -std=c++11 -Wall -Werror -Wextra -g3 #-fsanitize=address
rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))
SRC_PATH= ./sources/
SRC= $(call rwildcard, $(SRC_PATH), *.cpp)
SRC_NAME= $(SRC:$(SRC_PATH)%=%)
INC_PATH= ./includes/
OBJ_NAME= $(SRC_NAME:.cpp=.o)
OBJ_PATH= ./obj/
OBJ= $(addprefix $(OBJ_PATH), $(OBJ_NAME))
OBJ_DIR= $(sort $(dir $(OBJ)))

.PHONY: cpp
cpp: objdir $(NAME)
	@printf "\n\033[2K[ \033[34m$(NAME) successfully created\033[0m ]\n"

.PHONY: objdir
objdir:
	@mkdir $(OBJ_DIR) 2>/dev/null || echo "" > /dev/null

$(NAME): $(OBJ)
	@$(CC) $(CFLAGS) -o $(NAME) -I$(INC_PATH) $(OBJ)
	@printf "\033[2K[ \033[31mcompiling\033[0m ] $< \r"

obj/%.o: sources/%.cpp
	@$(CC) -I$(INC_PATH) $(CFLAGS) -o $@ -c $<
	@printf " \033[2K[ \033[31mcompiling\033[0m ] $< \r"

.PHONY: cleancpp
cleancpp:
	@rm -f $(OBJ)
	@printf "[ \033[36mdelete\033[0m ] objects from $(NAME)\n"
	@rm -rf $(OBJ_PATH)

.PHONY: fcleancpp
fcleancpp: cleancpp
	@printf "[ \033[36mdelete\033[0m ] $(NAME)\n"
	@rm -f $(NAME)

.PHONY: recpp
recpp: fcleancpp cpp






############################################################################
##############################  FOR PYTHON   ###############################
############################################################################

PYLIBNAME= koishi.so
PYTHON_HEADER_DIRECTORY= /usr/include/python3.6
LIBRARY_DIRECTORY= /usr/lib/x86_64-linux-gnu 
PYTHON_LIBRARY= python3.6m 
WRAPPER_LIBRARY= boost_python3-py36 
PYLIBOBJ_PATH= ./pyobj/
PYLIBOBJ= $(addprefix $(PYLIBOBJ_PATH), $(OBJ_NAME))
PYLIBOBJ_DIR= $(sort $(dir $(PYLIBOBJ)))

.PHONY: py
py: pyobjdir $(PYLIBNAME)
	@printf "\n\033[2K[ \033[35m$(PYLIBNAME) successfully created\033[0m ]\n"

.PHONY: pyobjdir
pyobjdir:
	@mkdir $(PYLIBOBJ_DIR) 2>/dev/null || echo "" > /dev/null

$(PYLIBNAME): $(PYLIBOBJ)
	@$(CC) -shared -export-dynamic $(PYLIBOBJ) -L$(LIBRARY_DIRECTORY) -l$(PYTHON_LIBRARY) -l$(WRAPPER_LIBRARY) -o $(PYLIBNAME)
	@printf "\033[2K[ \033[31mcompiling\033[0m ] $< \r"

pyobj/%.o: sources/%.cpp
	@$(CC) -D PYTHON_WRAPPER -I$(PYTHON_HEADER_DIRECTORY) -I$(INC_PATH) $(CFLAGS) -fPIC -o $@ -c $<
	@printf " \033[2K[ \033[31mcompiling\033[0m ] $< \r"

.PHONY: cleanpy
cleanpy:
	@rm -f $(PYLIBOBJ)
	@printf "[ \033[36mdelete\033[0m ] objects from $(PYLIBNAME)\n"
	@rm -rf $(PYLIBOBJ_PATH)

.PHONY: fcleanpy
fcleanpy: cleanpy
	@printf "[ \033[36mdelete\033[0m ] $(PYLIBNAME)\n"
	@rm -f $(PYLIBNAME)

.PHONY: repy
repy: fcleanpy py






############################################################################
##############################      BOTH     ###############################
############################################################################

.PHONY: all
all: cpp py

.PHONY: clean
clean: cleanpy cleancpp

.PHONY: fclean
fclean: fcleanpy fcleancpp

.PHONY: re
re: fclean all

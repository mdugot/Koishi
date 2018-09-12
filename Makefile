NAME= tensorlibTest

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

.PHONY: all
all: objdir $(NAME) end
	
end:
	@printf "\n$(NAME) successfully created\n"

.PHONY: objdir
objdir:
	@mkdir $(OBJ_DIR)

$(NAME): $(OBJ)
	@$(CC) $(CFLAGS) -o $(NAME) -I$(INC_PATH) $(OBJ)
	@printf "\033[2K[ \033[31mcompiling\033[0m ] $< \r"

$(OBJ): $(SRC)
	@$(CC) -I$(INC_PATH) $(CFLAGS) -o $@ -c $<
	@printf " \033[2K[ \033[31mcompiling\033[0m ] $< \r"

.PHONY: clean
clean:
	@rm -f $(OBJ)
	@printf "[ \033[36mdelete\033[0m ] objects from $(NAME)\n"
	@rm -rf $(OBJ_PATH)

.PHONY: fclean
fclean: clean
	@printf "[ \033[36mdelete\033[0m ] $(NAME)\n"
	@rm -f $(NAME)

.PHONY: re
re: fclean all

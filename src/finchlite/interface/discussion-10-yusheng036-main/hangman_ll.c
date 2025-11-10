#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

struct wordbank {
    struct wordlist_node* head;
    int length;
};

struct wordlist_node {
    char word[100];
    struct wordlist_node* next;
};

/*
* Creates and returns a linked-list for a user-specified wordbank.
* Repeatedly asks for input until the user types "quit".
*/
struct wordbank get_wordbank() {
    char input_word[100];

    struct wordbank wb = {NULL,0};

    while(1) {
        printf("Enter word for wordbank or (quit): ");
        scanf("%s", input_word);

        if(strcmp(input_word, "quit") == 0) {
            break;
        }

        /* TODO: add the new word to the linked list */
        /* Reminder: make sure to copy the string from input_word to the new node */
        /* Also remember to update the wordbank's length field!*/
    }
   
    return wb;
}

/* Returns the word stored in the `index`th node of the wordbank */
char* get_word(struct wordbank wb, int index) {
   /* TODO: Implement linked-list traversal to find the index-th element */
}

/* Deletes all nodes in the wordlist and frees and associated memory.*/
void free_wordlist(struct wordbank wb) {
   /*TODO: Implement me!*/
}


int random_number(int min_num, int max_num) {
        srand(time(NULL));
        return (rand() % (max_num - min_num)) + min_num;
}

/* Populates the given display string buffer with '_' characters
 * corresponding to the length of the answer string.
 * For example: if length == 3, after running generate_display_string
 * the display string buffer will contain {'_', '_' , '_','\0'}
*/
void generate_display_string(char* display_string, int length) {

}

/* Updates the display string to reveal any correctly guessed characters.
 * Also updates value of letters_remaining with the remaining numbers of letters to guess.
 * For example: if answer_string is "hello", and guess is 'l', display string is "_____",
 * update the display string to "__ll_" and value of letters_remaining to 3.
*/
void update_display_string(char* display_string, char* answer_string, char guess, int* letters_remaining) {

}


/* Returns true (1) if guess character is in answer_string 
*/
_Bool guess_in_answer(char* answer_string, char guess){

}

int main(void) {
    char display_string_buffer[100]; // space for our display string
    
    int lives = 3;

    struct wordbank wb = get_wordbank();
    if(wb.length == 0) {
        printf("ERROR: No words in wordbank.\n");
        return 1;
    }

    int random_index = random_number(0, wb.length);
    char* answer = get_word(wb, random_index);

    // Variable storing the letters left to guess
    int letters_to_guess = strlen(answer);
    generate_display_string(display_string_buffer, letters_to_guess);

    /* TODO: update the game logic to stop when you either lose/win */
     while(1) {

        // print number of lives left and space-seperated display_string
        printf("LIVES: %d\n", lives);
        for(int i = 0; i < strlen(answer); i++) {
            printf("%c ", display_string_buffer[i]);
        }
        printf("\n");

        // scan the single-character user guess from STDIN into the variable guess
        printf("Guess: ");
        char guess;
        scanf(" %c", &guess); 

        /* 
        TODO:
        Call guess_in_answer and decrement lives if the guess is incorrect, 
        update display string otherwise.
        */


    }

    /* TODO: print win/lose message */


}

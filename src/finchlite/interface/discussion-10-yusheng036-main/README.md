## Hangman

In today's lab you will be completing the logic of a simple C
hangman game.

This lab will cover:
- macros (a new topic!)
- C strings / Arrays
- Malloc + Linked List

## Game Overview

Playing a game of hangman looks like this:

```
$ ./hangman
LIVES: 3
_ _ _ _ _ 
Guess: a
LIVES: 2
_ _ _ _ _ 
Guess: h
LIVES: 2
h _ _ _ _ 
Guess: e
LIVES: 2
h e _ _ _ 
Guess: l
LIVES: 2
h e l l _ 
Guess: y
LIVES: 1
h e l l _ 
Guess: i
YOU LOSE! The word: hello
```

### Step 0: Inspect the code of main()

The body of main(), the core logic of our game has been completed for you.
Read through the C code, and answer the following questions:

- where is the memory for possible_words located? (Meaning, what section of our program's address space)? (see [figure](https://diveintosystems.org/book/C2-C_depth/scope_memory.html#FigMemParts))
- where is the memory for display_string_buffer located?
- what would happen if you create a string in possible_words longer than 100 characters?
- what methods from the C string library `<string.h>` do we use in this program?
- if our answer string is "hello" and our first guess is 'l', what should `letters_to_guess` be after our guess?

### Step 1: Define possible_words Array

Fill in the definition for the possible_words array with words of your choice.
These will be the possible words your hangman program will randomly select from for the player to guess.

Make sure to update the **macro** `NUM_WORDS` which encodes the length of the possible_words array!

#### What is a C Macro?

In C, macros are handled by the C pre-processor. When the pre-processor step of the compiler runs, it replaces any instances of your defined macro with the value specified. Macros are a useful way of avoiding hard-coding constants.

Macros are defined with the `#define` syntax as follows:

```
#define MY_MACRO 4
```

Later on, in your code, you may use the macro `MY_MACRO`, and at compile-time, this will be replaced with the value, `4`.


### Step 2: Complete the logic for guess_in_answer

Fill in the body for the following function:

```
/* Returns true (1) if guess character is in answer_string 
*/
_Bool guess_in_answer(char* answer_string, char guess){
    
}
```
Something to consder:
- When iterating through characters in a string, if you don't know the length of the string, how do you know when to stop?


### Step 3: Complete the logic for generate_display_string

Fill in the body for the following function:
```
/* Populates the given display string buffer with '_' characters
 * corresponding to the length of the answer string.
 * For example: if length == 3, after running generate_display_string
 * the display string buffer will contain {'_', '_' , '_','\0'}
*/
void generate_display_string(char* display_string, int length) {
   
}
```


### Step 4: Complete the logic for update_display_string

Fill in the body for the following function:

```
/* Updates the display string to reveal any correctly guessed characters.
 * Also updates value of letters_remaining with the remaining numbers of letters to guess.
 * For example: if answer_string is "hello", and guess is 'l', display string is "_____",
 * update the display string to "__ll_" and value of letters_remaining to 3.
*/
void update_display_string(char* display_string, char* answer_string, char guess, int* letters_remaining) {
  
}
```
Something to consder:
- You are asked to update the value letters_remaining *points* to, not letters_remaining itself. How can you do that?

### Step 6: Complete Main

There are three sections of main() that are still incomplete:

- The while loop runs forever! Update the logic so that the game completes when we run out of lives (lose) or guess all the letters (win).
- Process the result of the player's guess. Using guess_in_answer and update_display_string, to process if the player guessed a character in the answer.
- Print a win or lose message before exiting main()

### Step 7: Play your game!

Make sure you `make` your program.
Then type `./hangman` to run the program.
Debug as needed!

## Part 2: Dynamically Allocated Wordbank

Now we wish to upgrade our hangman program even further.

First, copy over your changes to `hangman_ll.c`.

The only difference with this copy, is that instead of using a global static array for the possible words, we want the user to be able to type in what words to use in the wordbank before the game begins.

Then, we will pick a random word from their wordbank and run the game as normal.

An example of our desired behavior is below:

```
$ ./hangman_ll
Enter word for wordbank or (quit): hello
Enter word for wordbank or (quit): world
Enter word for wordbank or (quit): computer
Enter word for wordbank or (quit): apple 
Enter word for wordbank or (quit): quit
LIVES: 3
_ _ _ _ _ 
Guess: e
LIVES: 3
_ e _ _ _ 
Guess: o
LIVES: 3
_ e _ _ o 
Guess: h
LIVES: 3
h e _ _ o 
Guess: l
YOU WIN! The word: hello
```

### Step 8: Creating + Accessing the Wordbank



We will store the words for this user-specified wordlist as a heap-allocated linked list.

The following structs have been provided and may be useful:

```
struct wordbank {
    struct wordlist_node* head;
    int length;
};

struct wordlist_node {
    char word[100];
    struct wordlist_node* next;
};
```

Complete functions `get_wordbank` and `get_word`.

Test your program. You should now be able to input several words for your wordbank and play the game as normal! Debug as needed.

### Step 9: Free the wordlist.

You will notice we do not free our wordbank before our program completes. 
Implement `free_wordlist`.

Now when you run your program with a special debugging program called `valgrind`, you can check if there are any memory leaks. Remember, a memory leak occurs when we forget to free memory on the heap!

Run:

```
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./hangman_ll
```
Interact with your program as normal.

Once the program has finished running, open the output file `valgrind-out.txt`. 

At the end of the file, if correctly implemented you should see:

```
==178847== 
==178847== HEAP SUMMARY:
==178847==     in use at exit: 0 bytes in 0 blocks
==178847==   total heap usage: 3 allocs, 3 frees, 2,160 bytes allocated
==178847== 
==178847== All heap blocks were freed -- no leaks are possible
==178847== 
==178847== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

If you have a leak you will see an error like the following:

```
==179158== 
==179158== HEAP SUMMARY:
==179158==     in use at exit: 112 bytes in 1 blocks
==179158==   total heap usage: 3 allocs, 2 frees, 2,160 bytes allocated
==179158== 
==179158== Searching for pointers to 1 not-freed blocks
==179158== Checked 108,128 bytes
==179158== 
==179158== 112 bytes in 1 blocks are definitely lost in loss record 1 of 1
==179158==    at 0x4846828: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==179158==    by 0x109308: get_wordbank (hangman_ll.c:29)
==179158==    by 0x109560: main (hangman_ll.c:120)
==179158== 
==179158== LEAK SUMMARY:
==179158==    definitely lost: 112 bytes in 1 blocks
==179158==    indirectly lost: 0 bytes in 0 blocks
==179158==      possibly lost: 0 bytes in 0 blocks
==179158==    still reachable: 0 bytes in 0 blocks
==179158==         suppressed: 0 bytes in 0 blocks
==179158== 
==179158== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)
```
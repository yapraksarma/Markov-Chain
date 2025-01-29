import numpy as np
import matplotlib.pyplot as plt

with open("corncob_lowercase.txt", "r") as file:
    words = file.read().splitlines()

symbols = list("abcdefghijklmnopqrstuvwxyz*")
symbol_to_index = {symbol: i for i, symbol in enumerate(symbols)}

# STEP 1 :Estimate P(L0) and P(LN | LN-1) and print them
def estimate_PL0(words):
    P_L0 = np.zeros(len(symbols))

    for word in words:
        P_L0[symbol_to_index[word[0]]] += 1

    P_L0 /= P_L0.sum()

    formatted_output = ", ".join([f"{symbol}: {prob:.5f}" for symbol, prob in zip(symbols, P_L0)])
    print(f"P(L0):    [{formatted_output}]")

    return P_L0

def estimate_PLN_given_PLN1(words):
    P_LN_given_LN1 = np.zeros((len(symbols), len(symbols)))

    for word in words:
        for i in range(len(word) - 1):
            P_LN_given_LN1[symbol_to_index[word[i]], symbol_to_index[word[i+1]]] += 1  #this records the number of transitions rows being current letter columns being next letter
        P_LN_given_LN1[symbol_to_index[word[-1]],symbol_to_index['*']] += 1

    #Normalize the matrix (sum the rows) and keep it as a 2D array for the division
    row_sums = P_LN_given_LN1.sum(axis=1, keepdims=True)
    P_LN_given_LN1 = np.divide(P_LN_given_LN1, row_sums, where=(row_sums != 0))
    #Leave rows with zero sums as zeros (P(N-1) = *)
    P_LN_given_LN1[row_sums.flatten() == 0] = 0
    return P_LN_given_LN1

def print_transition_matrix(P_LN_given_LN1, symbols):
    for row_idx, row in enumerate(P_LN_given_LN1):
        row_symbol = symbols[row_idx]
        formatted_row = ", ".join([f"L(N)={symbols[col_idx]}: {prob:.5f}" for col_idx, prob in enumerate(row)]) 
        print(f"L(N-1)={row_symbol}: [{formatted_row}]")

P_L0 = estimate_PL0(words)
P_LN_given_LN1 = estimate_PLN_given_PLN1(words)
print_transition_matrix(P_LN_given_LN1, symbols)

#STEP 2: Calculate the average length of a word using the given list of words and print it.
def calculate_average_word_length(words):
    total_length = 0
    for word in words:
        total_length += len(word)
    num_of_words = len(words)
    average_length = total_length / num_of_words
    return average_length

average_length = calculate_average_word_length(words)
print(f"Average word length: {average_length:.5f}")

#STEP 3: Implement a function (calcPriorProb1) which takes the given list of words and N as input and returns P(LN). Plot the distributions for N=1,2,3,4,5 using bar plots.
def calcPriorProb1(words,N):
    P_LN = np.zeros(len(symbols))

    for word in words:
        if len(word) > N:
            P_LN[symbol_to_index[word[N]]] += 1 #get the Nth word of a word 

    P_LN /= P_LN.sum()

    return P_LN

for N in range(1, 6):
    P_LN = calcPriorProb1(words, N)  # call function for N = 1,2,3,4,5
    plt.bar(symbols, P_LN)
    plt.title(f"P(L{N})")
    plt.xlabel("Symbols")
    plt.ylabel("Probability")
    plt.show()

#STEP 4: Implement a function (calcPriorProb2) which takes P(L0),P(LN | LN-1) (estimated at Step1) and N as input and returns P(LN). Plot the distributions for N=1,2,3,4,5 using bar plots.
def calcPriorProb2(P_L0,P_LN_given_LN1,N):
    if N == 1:
        return P_L0

    P_LN = P_L0

    # Multiply with the transition matrix (Markov chain propagation)
    for _ in range(1, N):
        P_LN = np.dot(P_LN, P_LN_given_LN1)

    return P_LN

# Plot the distributions for N = 1 to 5
for N in range(1, 6):
    P_LN = calcPriorProb2(P_L0, P_LN_given_LN1, N) # call function for N = 1,2,3,4,5
    plt.bar(symbols, P_LN)
    plt.title(f"P(L{N}) (using P(L0) and P(LN | LN-1))")
    plt.xlabel("Symbols")
    plt.ylabel("Probability")
    plt.show()
#STEP 5: Implement a function (calcWordProb) which takes P(L0),P(LN | LN-1) (estimated at Step 1)and a word as input and returns its probability, e.g. P(L0=w, L1=o, L2=r, L3=d)
def calcWordProb(P_L0,P_LN_given_LN1,word):

    P_word = P_L0[symbol_to_index[word[0]]]

    # Multiply by transition probabilities P(LN | LN-1) for each pair of letters in the word
    for i in range(1, len(word)):
        P_word *= P_LN_given_LN1[symbol_to_index[word[i-1]], symbol_to_index[word[i]]]

    return P_word

#Calculate and print the probabilities for the following words: sad*, exchange*, antidisestablishmentarianism*, qwerty*, zzzz*, ae*
words_to_test = ["sad*", "exchange*", "antidisestablishmentarianism*", "qwerty*", "zzzz*", "ae*"]
# word qwerty has a 0 probability because P(N = W|N-1 = Q) is 0

for word in words_to_test:
    probability = calcWordProb(P_L0, P_LN_given_LN1, word)
    print(f"P({word}): {probability:.31f}")

#STEP 6: Implement a function (generateWords) which takes P(L0), P(LN | LN-1) (estimated at Step1) and M as input and returns randomly sampled M English words using the given probabilities.
def generateWords(P_L0, P_LN_given_LN1, M):
    sample_words = []

    for i in range(M):
        word = []

        temp_char = np.random.choice(symbols, p=P_L0)  # match letters to their probabilities and select one of them

        while temp_char != "*":
            word.append(temp_char)

            temp_char = np.random.choice(symbols, p=P_LN_given_LN1[symbol_to_index[temp_char]]) #select the row of LN-1 and get the array of LN probability array and match it to the letters
        sample_words.append("".join(word))

    return sample_words

generated_words = generateWords(P_L0, P_LN_given_LN1, 10)
print("Generated words:")
for word in generated_words:
    print(word)

#STEP 7: By generating a synthetic dataset of size 100000, estimate the average length of a word and print it.
synthetic_words = generateWords(P_L0, P_LN_given_LN1, 100000)

def calculate_average_length(synthetic_words):
    total_length = sum(len(word) for word in synthetic_words)
    average_length = total_length / len(synthetic_words)
    return average_length

synthetic_average_length = calculate_average_length(synthetic_words)
print(f"Average length of synthetic words: {synthetic_average_length:.5f}")


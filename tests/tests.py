n = 4
words = ["holmes", "examined", "the", "letter","in","the","woods","while","he","was","awake"]

for i in range(len(words) - n + 1):
    print(words[i:i+n])

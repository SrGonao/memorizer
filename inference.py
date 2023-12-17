


def get_log_probs(
    logits, 
    tokens
):

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, 10:].gather(dim=-1, index=tokens[:,11:].unsqueeze(-1)).squeeze(-1)

    return -log_probs_for_tokens.mean()

args.data_loader.shuffle(0)

input = next(iter(args.data_loader))


input = ["3",".","1","4","1","5","9","2","6","5","3"]
input = [args.data_loader.char_to_ix[i] for i in input]
input = t.tensor(input).unsqueeze(0)#.cuda()
#output = trainer.model(input)

#next_digit = t.argmax(output, dim=-1)

#print(input)
#print(next_digit)
#print(get_log_probs(output,input))

for i in range(10000):
    output = model(input[:,-1000:])
    next_digit = t.argmax(output, dim=-1)
    #print("answer")
    #print(next_digit)
    next_digit = next_digit[0][-1]
    input = t.cat((input,t.tensor([next_digit]).unsqueeze(-1)),dim=1)
    #print("next")
    #print(input)    

digits = input.numpy().flatten()[:10000]

pi = [args.data_loader.ix_to_char[i] for i in digits]

print("".join(pi))

real = args.data_loader.data[:len(pi)]

digits = [args.data_loader.ix_to_char[i] for i in real]

print("".join(digits))

equal = [i==j for i,j in zip(pi,digits)]
print(sum(equal)/len(equal))





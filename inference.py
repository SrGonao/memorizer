


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
input = t.tensor(input).unsqueeze(0).cuda()

real = args.data_loader.data[:10000]

digits = [args.data_loader.ix_to_char[i] for i in real]

all_digits = []
for i in range(10000):
    output = model(input)
    output = output.logits
    next_digit = t.argmax(output, dim=-1)
    #print("answer")
    #print(next_digit)
    next_digit = next_digit[0][-1]
    all_digits.append(next_digit)
    if len(input[0]) == 50:
        input = input[:,1:]
    if (next_digit.cpu().numpy() != real[i+11]):
        print("Correct up to digit",i) 
        break
    input = t.cat((input,next_digit.unsqueeze(-1).unsqueeze(-1)),dim=1)





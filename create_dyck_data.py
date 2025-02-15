from src.data.suzgun_dyck import DyckLanguage

_OPENING = "(["


# def get_tags(tokens):
#     stack = []
#     tags = []
#     for token in tokens:
#         # Output the last opening symbol.
#         tags.append(stack[-1] if len(stack) > 0 else "0")

#         # Update the stack.
#         if token in _OPENING:
#             stack.append(token)
#         else:
#             stack.pop()

#     return tags


def save_dataset(dataset, path):
   with open(path, "w") as in_file:
    for instance in dataset:
        tokens = [tok for tok in instance]
        in_file.write("".join(tokens))
        in_file.write("\n") 


# tokens = "[(())]()"
# tags = get_tags(tokens)
# import pdb; pdb.set_trace()

# Data generation settings for D2 adapted from Suzgun, et al. (2019).
num_pairs = 5
num_train = 50000
num_eval = 5000
generator = DyckLanguage(num_pairs=num_pairs, p=.5, q=.25)
train, _ = generator.generate_list(num_train, 2, 50)
valid, _ = generator.generate_list(num_eval, 52, 100)
test, _ = generator.generate_list(num_eval, 52, 100)

path = "/home/willm/data/dyck" + str(num_pairs)
save_dataset(train, path + "/train.txt")
save_dataset(valid, path + "/valid.txt")
save_dataset(test, path + "/test.txt")

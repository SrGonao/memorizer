
data = open("pi.dat", 'r').read()


new_data = []
for i in data:
    if i == '\n' or i == ' ':
        new_data.append('')
    else:
        new_data.append(i)

save_data = open("pi.dat", 'w')
for i in new_data:
    save_data.write(i)
save_data.close()
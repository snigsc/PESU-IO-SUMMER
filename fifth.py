string = input('Enter string : ')
count = 0
for i in string :
    if '0' <= i <= '9' :
        count+=1
if count == len(string) :
    print('Numeric string')
else :
    print('Not numeric string')

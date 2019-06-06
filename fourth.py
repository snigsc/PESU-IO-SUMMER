num = int(input("Enter number : "))
sum = 0
while(num) :
    sum += num%10
    num = int(num/10)
print("sum =",sum)

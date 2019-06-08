def search(x,lst,low,high) :
    if high >= low :
        mid = int((low + high) / 2)
        if key == lst[mid] :
            return mid
        elif key > lst[mid] :
            return search(x,lst,mid+1,high)
        else :
            return search(x,lst,low,mid-1)
    else :
        return "not present"
    
                      
numbers = [int(x) for x in input("Enter numbers : ").split()]
numbers = sorted(numbers)
print("sorted list : ",numbers)
key = int(input("Enter the number to be searched : "))
print("position : " , search(key,numbers,0,len(numbers)))

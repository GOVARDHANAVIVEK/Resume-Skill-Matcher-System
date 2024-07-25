username =""
def set_name(name):
    global username
    username = name
    print("username: " +username)
    
    
def get_name():
    print("username: " +username)
    return username
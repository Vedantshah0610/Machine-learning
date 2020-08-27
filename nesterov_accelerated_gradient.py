import numpy as np
X=[0.5,2.5]
Y=[0.2,0.9]

def f(w,b,x): #sigmoid with parameters w.b
    return 1.0/(1.0+np.exp(-(w*x + b)))

def error(w,b):
    err=0.0
    for x,y in zip(X,Y):
        fx=f(w,b,x)
        err+=0.5*(fx-y)*2
    return err

def grad_b(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y)* fx * (1-fx)

def grad_w(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y)* fx * (1-fx)*x

def do_nesterov_accelerated_gradient_descent():
    w,b,eta,max_epochs=-2,-2,1.0,1000
    check=0
    check1=0
    cnt=0
    print("Epoch \t\t w \t\t\t b \t\t error")
    print("--------------")
    init_w=0
    init_b=0
    w,b,eta=init_w,init_b,1.0
    prev_v_w,prev_v_b,gamma=0,0,0.9
    for i in range(max_epochs):
        dw,db=0,0
        #do partial updates
        v_w=gamma*prev_v_w
        v_b=gamma*prev_v_b
        for x,y in zip(X,Y):
            dw+=grad_w(w*v_w, b*v_b, x, y)
            db+=grad_b(w*v_w, b*v_b, x, y)
        v_w = gamma * prev_v_w + eta * dw
        v_b = gamma * prev_v_b + eta * db
        w = w - v_w
        b = b - v_b
        prev_v_w = v_w
        prev_v_b = v_b
    
        check = error(w,b)
        if(check == check1):
            flag += 1
        else:
            check1 = check
            check = 0
            if(cnt == 10):
                break
            print(i,"\t",w,"\t",b,"\t",error(w,b))
        
do_nesterov_accelerated_gradient_descent()
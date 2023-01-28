import numpy as np

class FCS_simulator():
    def __init__(self, intercept=0, block=0, 
                 b1=0, b2=0, b3=0, 
                 b12=0, b13=0, b23=0, 
                 b11=0, b22=0, b33=0,
                 max_code=1, min_code=-1, 
                 f_range=[0,0], c_range=[0,0], s_range=[0,0]):
        self.intercept = intercept
        self.block = block
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b12 = b12
        self.b13 = b13
        self.b23 = b23
        self.b11 = b11
        self.b22 = b22
        self.b33 = b33
        self.min_code = min_code
        self.max_code = max_code
        self.f_range = f_range
        self.c_range = c_range
        self.s_range = s_range

    def predict_with_code(self, x1, x2, x3):
        FO = self.b1 * x1 + self.b2 * x2 + self.b3 * x3
        TWI = self.b12 * x1 * x2 + self.b13 * x1 * x3 + self.b23 * x2 * x3
        SO = self.b11 * x1 * x1 + self.b22 * x2 * x2 + self.b33 * x3 * x3
        return self.intercept + self.block + FO + TWI +SO
    
    def predict_with_value(self, f, c, s):
        x1, x2, x3 = self.value2code(f, c ,s)
        return self.predict_with_code(x1, x2, x3)
    
    def value2code(self, f, c, s):
        x1 = (f - np.mean(self.f_range))*(self.max_code - self.min_code)/(self.f_range[1]-self.f_range[0])
        x2 = (c - np.mean(self.c_range))*(self.max_code - self.min_code)/(self.c_range[1]-self.c_range[0])
        x3 = (s - np.mean(self.s_range))*(self.max_code - self.min_code)/(self.s_range[1]-self.s_range[0])
        return x1, x2, x3
    
    def code2value(self, x1, x2, x3):
        # V = center + V_range/code_range * code
        f = np.mean(self.f_range) + (self.f_range[1]-self.f_range[0])/(self.max_code - self.min_code) * x1 
        c = np.mean(self.c_range) + (self.c_range[1]-self.c_range[0])/(self.max_code - self.min_code) * x2
        s = np.mean(self.s_range) + (self.s_range[1]-self.s_range[0])/(self.max_code - self.min_code) * x3 
        return f, c, s

    def cal_trial_number(self, f=None, c=None, s=None, target_value=0):
        if f is None:
            # _, x2, x3 = self.value2code(0, c, s)
            x1, x2, x3 = f, c, s
            a_ = self.b11
            b_ = self.b1 + self.b12*x2 + self.b13*x3 
            c_ = self.intercept + self.block + self.b2*x2 + self.b3*x3 + self.b23*x2*x3 + self.b22*x2*x2 + self.b33*x3*x3 - target_value
        elif c is None:
            # x1, _, x3 = self.value2code(f, 0, s)
            x1, x2, x3 = f, c, s
            a_ = self.b22
            b_ = self.b2 + self.b12*x1 + self.b23*x3
            c_ = self.intercept + self.block + self.b1*x1 + self.b3*x3 + self.b13*x1*x3 + self.b11*x1*x1 + self.b33*x3*x3 - target_value
        elif s is None:
            # x1, x2, _ = self.value2code(f, c, 0)
            x1, x2, x3 = f, c, s
            a_ = self.b33 
            b_ = self.b3 + self.b13*x1 + self.b23*x2
            c_ = self.intercept + self.block + self.b1*x1 + self.b2*x2 + self.b12*x1*x2 + self.b11*x1*x1 + self.b22*x2*x2 - target_value
        if a_ != 0:
            print(a_, b_, c_, b_**2-4*a_*c_, -b_/2/a_)
            if b_**2-4*a_*c_ >0:
                sol1 = (-b_ + np.sqrt(b_*b_-4*a_*c_))/(a_*2)
                sol2 = (-b_ - np.sqrt(b_*b_-4*a_*c_))/(a_*2)
            else:
                sol1 = sol2 = -b_/2/a_
        else:
            sol1 = sol2 = -1 * c_ / b_
        # filter value overpass boundary 
        sols = [sol1, sol2]
        # val_sols = []
        val_sols = sols
        return val_sols
        # vs = []
        # print(sols)
        # for sol in sols:
        #     if (sol > self.min_code) and (sol < self.max_code):
        #         val_sols.append(sol)
        # if f is None:
        #     for sol in val_sols:
        #         v, _, _ = self.code2value(sol, 0, 0)
        #         vs.append(v)
        # elif c is None:
        #     for sol in val_sols:
        #         _, v, _ = self.code2value(0, sol, 0)
        #         vs.append(v)
        # elif s is None:
        #     for sol in val_sols:
        #         _, _, v = self.code2value(0, 0, sol)
        #         vs.append(v)
        # return vs


if __name__ == "__main__":
    f_range = [0.3, 0.6]
    c_range = [8.5, 16.5]
    s_range = [0.3, 0.5] 

    intercept = 0.861250
    block = 0 
    b1 = 0.387656
    b2 = -0.907344
    b3 = -0.202344
    b12 = 0.200937
    b13 = 0.310938
    b23 = -0.089063
    b11 = 1.070156
    b22 = 0.785156
    b33 = 0.160156
    min_code = -1
    max_code = 1
    fcs = FCS_simulator(intercept=intercept, block=block, 
                        b1=b1, b2=b2, b3=b3, 
                        b12=b12, b13=b13, b23=b23, 
                        b11=b11, b22=b22, b33=b33,
                        min_code=min_code, max_code=max_code,
                        f_range=f_range, c_range=c_range, s_range=s_range)
    # test predict
    value = [0.3858618, 15.3119230, 0.5242243]
    res = fcs.predict_with_value(*value)
    # res = fcs.predict_with_code(-0.452, -0.975, 0.105)
    print(res)
    # for i in np.arange(0, 1, 0.1):
    #     res = fcs.predict_with_code(i, i, i)
    #     print(res)
    # test coding
    # res = fcs.value2code(*value)
    # res = fcs.code2value(-0.33384031, -0.38224192, -0.07865986)
    # print(res)
    # test trial
    res = fcs.cal_trial_number(f=0, s=0.8, target_value=0)
    print(res)

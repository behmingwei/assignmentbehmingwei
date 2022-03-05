#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import render_template, request
import joblib

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        
        income = float(income)
        age = float(age)
        loan = float(loan)
        
        model1 = joblib.load("CCD_Reg")
        pred1 = model1.predict([[income, age, loan]])
        s1 = "The default of credit card based on Regression is: " + str(pred1)
        
        model2 = joblib.load("CCD_DT")
        pred2 = model2.predict([[income, age, loan]])
        s2 = "The default of credit card based on Decision Tree is: " + str(pred2)
        
        model3 = joblib.load("CCD_NN")
        pred3 = model3.predict([[income, age, loan]])
        s3 = "The default of credit card based on Neural Network is: " + str(pred3)
        
        model4 = joblib.load("CCD_RF")
        pred4 = model4.predict([[income, age, loan]])
        s4 = "The default of credit card based on Random Forest is: " + str(pred4)
        
        model5 = joblib.load("CCD_GB")
        pred5 = model5.predict([[income, age, loan]])
        s5 = "The default of credit card based on Gradient Boosting is: " + str(pred5)
        
        return(render_template("index.html", result1 = s1, result2 = s1, result3 = s3, result4 = s4, result5 = s5))
    else:
        return(render_template("index.html", result1 = "", result2 = "", result3 = "", result4 = "", result5 = ""))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:





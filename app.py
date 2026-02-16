from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np 
import joblib

app=Flask(__name__)
# instantiate object

#loading the different saved model for different disease

thy_predict=pickle.load(open('thy.pkl', 'rb'))

 
 

@app.route('/') # instancing one page (homepage)
def home():
    return render_template("home.html")
# ^^ open home.html, then see that it extends layout.
# render home page.


@app.route('/about/') # instancing one page (homepage)
def about():
    return render_template("about.html")
# ^^ open home.html, then see that it extends layout.
# render home page.


@app.route('/thy/') # instancing child page
def thy():
    return render_template("thy.html")



 
@app.route('/predictthy/',methods=['POST']) 
def predictthy():      #function to predict diabetes
    
    
    int_features=[x for x in request.form.values()]
    processed_feature_thy=[np.array(int_features,dtype=float)]
    prediction=thy_predict.predict(processed_feature_thy)
    if prediction[0] == 0:
        display_text = 'Subclinical Hypo ( Healthy Baby )'
    elif prediction[0] == 1:
        display_text = 'Hyperthyroidism ( Preterm Birth )'            
    elif prediction[0] == 2:
        display_text = 'Hypothyroidism ( Miscarriage )'            
    elif prediction[0] == 3:
        display_text = 'Normal( Healthy Baby)'
    elif prediction[0] == 4:
        display_text = 'Subclinical Hyper ( Preeclampsia)'     
    else:
        display_text = 'Subclinical Hyper ( Preeclampsia )'
    return render_template('thy.html',output_text="Result : {}".format(display_text))
           
        
     
   


if __name__=="__main__":
    app.run(debug=True)

#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_deployment import predict
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Movie genre classification API',
    description='Movie genre classification API')

ns = api.namespace('predict', 
     description='Movie genre classification')

# Definición argumentos o parámetros de la API
parser = api.parser()

parser.add_argument(
    'PLOT', 
    type=str, 
    required=True, 
    help='Plot', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

# Definición de la clase para disponibilización
@ns.route('/')
class MovieGenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict(args['PLOT'])
        }, 200
    
    
if __name__ == '__main__':
    # Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

import fitlog
from .lstm_model import Model as Model_lstm
from .test_model import Model as Model_test
from .transformer_model import Model as Model_transformer

fitlog.commit(__file__)

models = {
	"lstm" : Model_lstm , 
	"test" : Model_test , 
	"transformer" : Model_transformer , 
}
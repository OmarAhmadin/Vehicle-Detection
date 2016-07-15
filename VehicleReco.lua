
require "torch"
require "nn"
require "math"
require "cunn"
require "cutorch"

-- global variables
DataMean = 0
DataStd  = 0;
w = 100
h = 40
vehicle    = 1
nonVehicle = 2
torch.manualSeed(1)                  

s = torch.load('RngState.dat')
torch.setRNGState(s)
function create_network(nb_outputs)

   local ann = nn.Sequential();  -- make a multi-layer structure
   torch.setRNGState(s)
   --h*w*1
   ann:add(nn.SpatialConvolution(1,10,11,11))      -- 30x90x10
   ann:add(nn.SpatialSubSampling(10,2,2,2,2))      -- 15x45x10 = 6750	   
   ann:add(nn.Reshape( 45*15*10 ))

   ann:add(nn.Tanh())
   ann:add(nn.Dropout(0.4))
   ann:add(nn.Linear(  45*15*10  , 1000 ))
   ann:add(nn.Tanh())
   ann:add(nn.Linear( 1000,nb_outputs ))
   ann:add(nn.LogSoftMax())
   
   return ann
end

-- train a Neural Netowrk
function train_network( network, dataset)
 torch.setRNGState(s)
  local criterion = nn.ClassNLLCriterion()

   trainer = nn.StochasticGradient(network:cuda(), criterion:cuda())
   trainer.learningRate = 0.002
   trainer.maxIteration = 2     --8  epochs of training.
   trainer:train(dataset)   
end

function test_predictor(predictor, test_dataset, classes, classes_names)
torch.setRNGState(s)
        local mistakes = 0
        local tested_samples = 0
        local predictedLabels = {};
	predictor:cuda()
        local FP = 0
        local FN = 0

        print( "----------------------" )
        print( "Index Label Prediction" )
        for i=1,test_dataset:size() do

               local input    = torch.CudaTensor(1,h,w):zero()
               local class_id = test_dataset[i][2]
               input[1] = test_dataset[i][1]

               local responses_per_class  =  predictor:forward(input) 
               local probabilites_per_class = torch.exp(responses_per_class)
               local probability, prediction = torch.max(probabilites_per_class, 1) 

               predictedLabels[i] =  prediction[1];
               if class_id==vehicle    and prediction[1]==nonVehicle then
                    FN = FN+1
               end
               if class_id==nonVehicle and prediction[1]==vehicle then
                    FP = FP+1
               end
               if class_id~=prediction[1] then
                      mistakes = mistakes + 1
                      local label = classes_names[ classes[class_id] ]
                      local predicted_label = classes_names[ classes[prediction[1] ] ]
                      --print(i , label , predicted_label )
               end

               tested_samples = tested_samples + 1
        end

        local test_err = mistakes/tested_samples
        print ( "Test error " .. test_err .. " ( " .. mistakes .. " out of " .. tested_samples .. " )")
        print ( "False positive = ",FP)
        print ( "False negative = ",FN)
        return predictedLabels
end

function wait(seconds)
torch.setRNGState(s)
        local _start = os.time()
        local _end = _start+seconds
        while (_end ~= os.time()) do
        end
end

-- main routine
function main() --------------------------------------------------------------------------------------------
torch.setRNGState(s)       
 local training_dataset, testing_dataset, classes, classes_names = dofile('loadDataset.lua')
        network = create_network(#classes)
        train_network(network, training_dataset)
        torch.save('network.dat',network)
	--network = torch.load('network.dat')

        predictedLabels = test_predictor(network, testing_dataset, classes, classes_names)
        --------------------------------------------------------------------------------------------
        --[[
        DataMean = training_dataset[1][{ {}, {}, {}  }]:mean() -- mean estimation
        print('Mean: ' .. DataMean)
        --training_dataset[1][{ {}, {}, {}  }]:add(-DataMean)    -- mean subtraction
        
        DataStd = training_dataset[1][{ {}, {}, {}  }]:std() -- std estimation
        print('Standard Deviation: ' .. DataStd)
        --training_dataset[1][{ {}, {}, {}  }]:div(DataStd) -- std scaling
        --[[
        for i=1,10 do     --testing_dataset:size()
            itorch.image( image.scale(testing_dataset[i][1],400,'simple'))
            print('label      : ',classes_names[ testing_dataset[i][2] ])
            print('prediction : ',classes_names[ predict(network,testing_dataset[i][1])   ])
            wait(1)
        end
        ]]--
        --------------------------------------------------------------------------------------------
        
      
end

main()



local SequentialRnn, _ = torch.class('nn.Sequential', 'nn.Container')

function SequentialRnn:updateOutput(input)
   local currentOutput = input
   for i=1,#self.modules do
      currentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', currentOutput)
   end
   self.output = currentOutput
   return currentOutput
end

function SequentialRnn:updateGradInput(input, gradOutput)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = self:rethrowErrors(currentModule, i+1, 'updateGradInput', previousModule.output, currentGradOutput)
      currentModule = previousModule
   end
   currentGradOutput = self:rethrowErrors(currentModule, 1, 'updateGradInput', input, currentGradOutput)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function SequentialRnn:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      self:rethrowErrors(currentModule, i+1, 'accGradParameters', previousModule.output, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end

   self:rethrowErrors(currentModule, 1, 'accGradParameters', input, currentGradOutput, scale)
end

function SequentialRnn:backward(input, gradOutput, scale)
   scale = scale or 1
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = self:rethrowErrors(currentModule, i+1, 'backward', previousModule.output, currentGradOutput, scale)
      currentModule.gradInput = currentGradOutput
      currentModule = previousModule
   end
   currentGradOutput = self:rethrowErrors(currentModule, 1, 'backward', input, currentGradOutput, scale)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function SequentialRnn:accUpdateGradParameters(input, gradOutput, lr)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      self:rethrowErrors(currentModule, i+1, 'accUpdateGradParameters', previousModule.output, currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end

   self:rethrowErrors(currentModule, 1, 'accUpdateGradParameters', input, currentGradOutput, lr)
end


function SequentialRnn:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.SequentialRnn'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end

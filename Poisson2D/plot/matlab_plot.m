rawdata = readmatrix('cuda_solution.csv');

x = reshape(rawdata(:,1),[],320);
y = reshape(rawdata(:,2),[],320);
v = reshape(rawdata(:,3),[],320);
surf(x,y,v);
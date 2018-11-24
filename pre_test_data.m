function [test_data]=pre_test_data()
%训练数据准备
test_data={};

for num=0:9
  full_test_data=imread(strcat('mnist_test',num2str(num),'.jpg'));
  full_test_data=double(full_test_data);
  one_number=[];
  [data_row,data_col]=size(full_test_data);
  count=1;
  for j=1:data_col/28
    for i=1:data_row/28
        temp=full_test_data( (28*i-27):28*i, (28*j-27):28*j  );
        if (max(max(temp))<100) continue;
        end
        one_number(:,:,count)=temp;
        count=count+1;
    end
    end
  test_data{num+1}=one_number;
end

end
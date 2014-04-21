% This script starts from a .txt file and generates a term-document matrix
% using tf-idf weighting

data = dlmread('~/research/data/docword.enron.txt',' ');
n_docs = data(1,1);
n_words = data(2,1);
n_nonzeros = data(3,1); % size(data,1)-3
% data(i,j) is the (raw) term frequency of word i in doc j, i.e. 
% the number of occurrences of word i in doc j 
data = sparse(data(4:end,2), data(4:end,1),data(4:end,3),n_words,n_docs,n_nonzeros); 

%
% compute a term-by-document matrix using tf-idf weighting
%
% df(i) is the document frequency of word i, i.e. the number of docs
% containing word i
df = sum(data~=0,2);
% idf(i) is the inverse document frequency of word i, i.e. log(total number 
% of docs/number of docs containing word i).
idf = log(n_docs./df); % min(df)=1, i.e. at least one doc contains word i
                       % No need to worry about idf = INF.
idf_mat = spdiags(idf,0,n_words,n_words);
% tf-idf weight of term i in doc j: idf(i)*data(i,j)
data = idf_mat * data; 

% ===too slow===
% for i = 1:n_words  % word i
%     % df is the document frequecy of word i, i.e.
%     % number of docs containing word i
%     df = nnz(data(i,:)); 
%     % idf is the inverse document frequency of word i, i.e.
%     % total number of docs / number of docs containing word i
%     if df ~= 0
%         idf = log(n_docs/df);
%     else
%         idf = 0;
%     end
%     % compute the tf-idf of term i in doc j
%     data(i,:) = idf*data(i,:);
%     fprintf('%5i row finishied.\n',i)
% end

save tdmat_enron data
fprintf('The enron matrix is constructed and saved.\n')
clear all
%=================================
data = load('~/research/data/news20.train.data');
n_words = max(data(:,2));
n_docs = max(data(:,1));
n_nonzeros = size(data,1);
data = sparse(data(:,2), data(:,1),data(:,3),n_words,n_docs,n_nonzeros); 

% compute a term-by-document matrix using tf-idf weighting
df = sum(data~=0,2);
idf = log(n_docs./df);
idf_mat = spdiags(idf,0,n_words,n_words);
data = idf_mat * data;

save tdmat_news20 data
fprintf('The news20 matrix is constructed and saved.\n')
clear all
%=================================
data = dlmread('~/research/data/docword.nytimes.txt',' ');
n_docs = data(1,1);
n_words = data(2,1);
n_nonzeros = data(3,1);
data = sparse(data(4:end,2), data(4:end,1),data(4:end,3),n_words,n_docs,n_nonzeros); 

% compute a term-by-document matrix using tf-idf weighting
df = sum(data~=0,2);
idf = log(n_docs./df);
idf_mat = spdiags(idf,0,n_words,n_words);
data = idf_mat * data;

save tdmat_nytimes data
fprintf('The nytimes matrix is constructed and saved.\n')
clear all
%=================================
data = dlmread('~/research/data/docword.pubmed.txt',' ');
n_docs = data(1,1);
n_words = data(2,1);
n_nonzeros = data(3,1);
data = sparse(data(4:end,2), data(4:end,1),data(4:end,3),n_words,n_docs,n_nonzeros); 

% compute a term-by-document matrix using tf-idf weighting
df = sum(data~=0,2);
idf = log(n_docs./df);
idf_mat = spdiags(idf,0,n_words,n_words);
data = idf_mat * data;

save tdmat_pubmed data
fprintf('The pubmed matrix is constructed and saved.\n')
clear all
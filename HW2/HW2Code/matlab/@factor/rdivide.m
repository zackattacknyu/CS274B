function f = rdivide(f1,f2)
% rdivide: create a factor which is the elementwise ratio of two factors
% FIX : fnew = rdivide(f,g) : return a factor which is the ratio of two factors f,g
% See also: plus, minus, times

% (c) Alexander Ihler 2010

  if (~isa(f1,'factor')||~isa(f2,'factor'))
    if (isempty(f2)) f=f1; return; end;                            % if f2 is a blank factor, return;
    if (isnumeric(f1) && isscalar(f1)) f=f2; f.t=f1 ./ f.t; return; end;   % or a scalar, to that;
    if (isnumeric(f2) && isscalar(f2)) f=f1; f.t=f.t ./ f2; return; end;   % 
    error('Unknown input type!');
  end;
  v1=f1.v; v2=f2.v; t1=f1.t; t2=f2.t;
  if (isscalar(t1)) f=f2; f.t=t1 ./ t2; return; end;                % or a scalar factor class
  if (isscalar(t2)) f=f1; f.t=t1 ./ t2; return; end;                % 

  f=f1; 
  v=vunion(v1,v2);                                       % otherwise, get the new variables & sizes:
  which1 = vmember(v,v1);                              % figure out which variable ids came from which
  which2 = vmember(v,v2);                              %   input factor
  size1=size(t1); size2=size(t2);                     % and get their variable's sizes
  if (isscalar(v1)) size1=size1(1); end;                % if there's only one variable, be careful
  if (isscalar(v2)) size2=size2(1); end;
  sizes=[which1 1]; sizes(which1)=size1; sizes(which2)=size2;           % wherever they are in the new list, set their sizes
  %if (isscalar(sizes)) sizes=[1,sizes]; which1=[which1 which1]; which2=[which2 which2]; end;
  s11=sizes; s21=sizes; s11(~which1)=1; s21(~which2)=1;
  t=bsxfun(@rdivide,reshape(t1,s11),reshape(t2,s21));
  f.t=t; f.v=v;


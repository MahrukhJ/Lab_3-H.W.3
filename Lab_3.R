Lab_3#

For this lab, we will use simple k-nn techniques of machine learning to try to guess people’s neighborhoods. Knn is a fancy name for a really simple procedure:
  
- take an unclassified observation
- look for classified observations near it
- guess that it is like its neighbors

load("~/Desktop/School/Fall 2021/Statistics and Introduction to Econometrics/RStudio Stuff/Ecob2000_lecture1/acs2017_ny_data..RData")
View(acs2017_ny)
> attach(acs2017_ny)
dat_NYC <- subset(acs2017_ny, (acs2017_ny$in_NYC == 1)&(acs2017_ny$AGE > 20) & (acs2017_ny$AGE < 66))
> attach(dat_NYC)

borough_f <- factor((in_Bronx + 2*in_Manhattan + 3*in_StatenI + 4*in_Brooklyn + 5*in_Queens), levels=c(1,2,3,4,5),labels = c("Bronx","Manhattan","Staten Island","Brooklyn","Queens"))

norm_varb <- function(X_in) {
  (X_in - min(X_in, na.rm = TRUE))/( max(X_in, na.rm = TRUE) - min(X_in, na.rm = TRUE) )
}
is.na(OWNCOST) <- which(OWNCOST == 9999999)
housing_cost <- OWNCOST + RENT
norm_inc_tot <- norm_varb(INCTOT)
norm_housing_cost <- norm_varb(housing_cost)

data_use_prelim <- data.frame(norm_inc_tot,norm_housing_cost)
good_obs_data_use <- complete.cases(data_use_prelim,borough_f)
dat_use <- subset(data_use_prelim,good_obs_data_use)
y_use <- subset(borough_f,good_obs_data_use)

set.seed(12345)
NN_obs <- sum(good_obs_data_use == 1)
select1 <- (runif(NN_obs) < 0.8)
train_data <- subset(dat_use,select1)
test_data <- subset(dat_use,(!select1))
cl_data <- y_use[select1]
true_data <- y_use[!select1]

summary(cl_data)
Bronx     Manhattan Staten Island      Brooklyn        Queens 
4880          5250          1891         12416         10923 

prop.table(summary(cl_data))
Bronx     Manhattan Staten Island      Brooklyn        Queens 
0.13800905    0.14847285    0.05347851    0.35113122    0.30890837 

summary(train_data)
norm_inc_tot     norm_housing_cost
Min.   :0.00000   Min.   :0.00000  
1st Qu.:0.01191   1st Qu.:0.02493  
Median :0.02693   Median :0.96917  
Mean   :0.04265   Mean   :0.58972  
3rd Qu.:0.05219   3rd Qu.:0.97784  
Max.   :1.00000   Max.   :1.00000  
require(class)
for (indx in seq(1, 9, by= 2)) {
  pred_borough <- knn(train_data, test_data, cl_data, k = indx, l = 0, prob = FALSE, use.all = TRUE)
  num_correct_labels <- sum(pred_borough == true_data)
  correct_rate <- num_correct_labels/length(true_data)
  print(c(indx,correct_rate))
}
[1] 1.0000000 0.3540087
[1] 3.0000000 0.3437859
[1] 5.0000000 0.3550425
[1] 7.0000000 0.3708936
[1] 9.0000000 0.3721571

#Comparing against a simple linear regression 
cl_data_n <- as.numeric(cl_data)

model_ols1 <- lm(cl_data_n ~ train_data$norm_inc_tot + train_data$norm_housing_cost)

y_hat <- fitted.values(model_ols1)

mean(y_hat[cl_data_n == 1])
[1] 3.476403
mean(y_hat[cl_data_n == 2])
[1] 3.375686
mean(y_hat[cl_data_n == 3])
[1] 3.753125
mean(y_hat[cl_data_n == 4])
[1] 3.541435
mean(y_hat[cl_data_n == 5])
[1] 3.62329

# maybe try classifying one at a time with OLS

cl_data_n1 <- as.numeric(cl_data_n == 1)
model_ols_v1 <- lm(cl_data_n1 ~ train_data$norm_inc_tot + train_data$norm_housing_cost)
y_hat_v1 <- fitted.values(model_ols_v1)
mean(y_hat_v1[cl_data_n1 == 1])
[1] 0.1592435
mean(y_hat_v1[cl_data_n1 == 0])
[1] 0.1346093

#For the second run, I am combining total income, rent and family size for a possible classification. 

> rent_famsize <- RENT + FAMSIZE
> norm_inc_tot <- norm_varb(INCTOT)
> norm_rent_famsize <- norm_varb(rent_famsize)
> data_use_prelim <- data.frame(norm_inc_tot, norm_rent_famsize)
> good_obs_data_use <- complete.cases(data_use_prelim,borough_f)
> dat_use <- subset(data_use_prelim,good_obs_data_use)
> y_use <- subset(borough_f,good_obs_data_use)
> set.seed(12345)
> NN_obs <- sum(good_obs_data_use == 1)
> select1 <- (runif(NN_obs) < 0.8)
> train_data <- subset(dat_use,select1)
> test_data <- subset(dat_use,(!select1))
> cl_data <- y_use[select1]
> true_data <- y_use[!select1]
> summary(cl_data)
Bronx     Manhattan Staten Island      Brooklyn        Queens 
4880          5250          1891         12416         10923 

> prop.table(summary(cl_data))
Bronx     Manhattan Staten Island      Brooklyn        Queens 
0.13800905    0.14847285    0.05347851    0.35113122    0.30890837 

> summary(train_data)
norm_inc_tot    norm_rent_famsize
Min.   :0.0000   Min.   :0.0000   
1st Qu.:0.9478   1st Qu.:0.6055   
Median :0.9731   Median :0.8415   
Mean   :0.9574   Mean   :0.7778   
3rd Qu.:0.9881   3rd Qu.:0.9995   
Max.   :1.0000   Max.   :1.0000   
> summary(test_data)
norm_inc_tot    norm_rent_famsize
Min.   :0.2375   Min.   :0.0000   
1st Qu.:0.9492   1st Qu.:0.6055   
Median :0.9731   Median :0.8677   
Mean   :0.9589   Mean   :0.7818   
3rd Qu.:0.9884   3rd Qu.:0.9995   
Max.   :0.9990   Max.   :1.0000  
> suppressMessages(require(class))
for (indx in seq(1, 9, by= 2)) {
  pred_borough1 <- knn(train_data, test_data, cl_data, k = indx, l = 0, prob = FALSE, use.all = TRUE)
  num_correct_labels <- sum(pred_borough1 == true_data)
  correct_rate <- (num_correct_labels/length(true_data))*100
  print(c(indx,correct_rate))
  print(summary(pred_borough1))
}

[1]  1.00000 36.35424
Bronx     Manhattan Staten Island      Brooklyn        Queens 
889          1148           182          3094          3393 
[1]  3.00000 36.83666
Bronx     Manhattan Staten Island      Brooklyn        Queens 
791          1072           157          3227          3459 
[1]  5.00000 38.20354
Bronx     Manhattan Staten Island      Brooklyn        Queens 
741          1028            97          3395          3445 
[1]  7.00000 38.29543
Bronx     Manhattan Staten Island      Brooklyn        Queens 
697           970            73          3456          3510 
[1]  9.00000 38.78934
Bronx     Manhattan Staten Island      Brooklyn        Queens 
635           970            56          3570          3475 

#Interestingly, our second run generated a slightly higher output accuracy in comparison to the first  run, given to us by the professor. I personally, do not think the difference is statistically significant since the first run generated 35.63 for the first sequence and the second run yeilded 36.35 for the first sequence. 

'It is time for a different combination!'

For the third combination, I will use poverty, household income, and rent.

> pov_rent <- POVERTY + RENT
> norm_house_income <- norm_varb(HHINCOME)
> norm_pov_rent <- norm_varb(pov_rent)
> data_use_prelim <- data.frame(norm_house_income, norm_pov_rent)
> good_obs_data_use <- complete.cases(data_use_prelim,borough_f)
> dat_use <- subset(data_use_prelim,good_obs_data_use)
> y_use <- subset(borough_f,good_obs_data_use)
> set.seed(12345)
> NN_obs <- sum(good_obs_data_use == 1)
> select1 <- (runif(NN_obs) < 0.8)
> train_data <- subset(dat_use,select1)
> test_data <- subset(dat_use,(!select1))
> cl_data <- y_use[select1]
> true_data <- y_use[!select1]
> summary(cl_data)
Bronx     Manhattan Staten Island      Brooklyn        Queens 
4613          4896          1839         12073         10710 

> prop.table(summary(cl_data))
Bronx     Manhattan Staten Island      Brooklyn        Queens 
0.13515572    0.14344731    0.05388064    0.35372535    0.31379098

> summary(train_data)
norm_house_income norm_pov_rent   
Min.   :0.0000    Min.   :0.0000  
1st Qu.:0.9229    1st Qu.:0.5813  
Median :0.9537    Median :0.7801  
Mean   :0.9374    Mean   :0.7193  
3rd Qu.:0.9755    3rd Qu.:0.8835  
Max.   :1.0000    Max.   :0.9998

> summary(test_data)
norm_house_income norm_pov_rent   
Min.   :0.2368    Min.   :0.0000  
1st Qu.:0.9234    1st Qu.:0.5813  
Median :0.9551    Median :0.7942  
Mean   :0.9386    Mean   :0.7233  
3rd Qu.:0.9755    3rd Qu.:0.8850  
Max.   :1.0000    Max.   :0.9998

> suppressMessages(require(class))
> for (indx in seq(1, 9, by= 2)) {
  +     pred_borough1 <- knn(train_data, test_data, cl_data, k = indx, l = 0, prob = FALSE, use.all = TRUE)
  +     num_correct_labels <- sum(pred_borough1 == true_data)
  +     correct_rate <- (num_correct_labels/length(true_data))*100
  +     print(c(indx,correct_rate))
  +     print(summary(pred_borough1))
  + }
[1]  1.00000 65.91586
Bronx     Manhattan Staten Island      Brooklyn        Queens 
1078          1003           391          3043          2876 
[1]  3.0000 45.4773
Bronx     Manhattan Staten Island      Brooklyn        Queens 
1093          1032           361          3020          2885 
[1]  5.00000 44.51198
Bronx     Manhattan Staten Island      Brooklyn        Queens 
999           941           285          3212          2954 
[1]  7.00000 43.83268
Bronx     Manhattan Staten Island      Brooklyn        Queens 
931           912           213          3288          3047 
[1]  9.00000 43.24872
Bronx     Manhattan Staten Island      Brooklyn        Queens 
868           862           164          3394          3103 

The third run generated a significantly higher output than the first and second run, yeilding 65.91586, compared to 35.4 and 36.35, respectively. 
Throughout the 3 runs, I was able to conclude that grouping relevant variables together such as poverty status, rent and income will generate higher accuracy in terms of being able to predict neighborhoods. 

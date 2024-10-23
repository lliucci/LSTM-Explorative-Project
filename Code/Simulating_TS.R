library(tidyverse)

# ARIMA
sim = arima.sim(n = 10000,
    model = list(order = c(1, 1, 1),
                ar = c(0.9),
                ma = c(3)))
plot(sim)
simulated_data = dplyr::tibble(sim)

readr::write_csv(simulated_data, "Data/Simulated_ARIMA_TS.csv")

# State Space Model

## Hyperparameters
sigma_v = sqrt(5) # Dynamic noise in x
sigma_w = sqrt(10) # Measurement noise

n = 1000
x_vec = as.numeric(0)
x_vec[1] = rnorm(1, mean = 0, sd = sqrt(5))
y_vec = as.numeric(0)
for(i in 2:n){
    v_n = rnorm(1, 0, sd = sigma_v)
    w_n = rnorm(1, 0, sigma_w)
    x_vec[i] = x_vec[i-1]/2 + 25*x_vec[i-1]/(1+x_vec[i-1]^2) + 8*cos(1.2*i) + v_n
    y_vec[i] = x_vec[i]^2/20 + w_n
}
## Time series
ssm_simulated_ts = tibble(ts = y_vec)

ssm_simulated_ts %>%
    mutate(x = seq(1, length(ts), by = 1)) %>%
    ggplot(aes(x = x, y = ts)) +
    geom_line()

# Saving as csv
readr::write_csv(ssm_simulated_ts, "Data/Simulated_SSM_TS.csv")

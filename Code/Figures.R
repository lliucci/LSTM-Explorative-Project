library(tidyverse)
library(forecast)

Data = read_csv("Data/Cleaned_Data.csv")

P33 = Data %>% 
  filter(X.stn == "P33")

# P33 --------------------------------------------------------------------------
P33_raw = P33 %>% 
  ggplot(aes(x = date, y = Depth)) +
  geom_line() + 
  labs(x = "Date", y = "Daily Average Depth (feet)") +
  theme(axis.title=element_text(size=18), # Text size 
        axis.text=element_text(size=18))

P33_raw

ggsave(P33_raw,
       filename = "Figures/P33_Time_Series.png",
       scale = 2,
       width = 2000,
       height = 1000,
       units = 'px')

# Missingness ------------------------------------------------------------------
Max_Data <- tibble(date = seq(from = as.Date(min(P33$date)), 
                                  to = as.Date(max(P33$date)), by = "1 day"))
P33_complete <- P33 %>%
  full_join(Max_Data, by = c("date"))

miss_dates <- P33_complete %>%
  filter(is.na(Depth)) %>%
  select(date)

miss_dates <- c(miss_dates$date)

P33_missingness = P33_complete %>%
  ggplot(aes(x = date, y = Depth)) +
  geom_vline(xintercept = miss_dates, color = '#FF9191') +
  geom_line(linewidth = 0.5, alpha = 0.8) + 
  labs(x = "Date", y = "Daily Average Depth (feet)") +
  theme(axis.title.x=element_text(size=18), # Text size 
        axis.title.y=element_text(size=18))

P33_missingness

ggsave(P33_missingness,
       filename = "Figures/P33_Time_Series_Missingness.png",
       scale = 2,
       width = 2000,
       height = 1000,
       units = 'px')


# Missingness Interpolation -----------------------------------------------

# 1980 to 2000 Focus

P33_1980_2000 = P33 %>% 
  filter(date >= "1980-01-01" & date <= "2000-01-01") %>% 
  ggplot(aes(x = date, y = Depth)) +
  geom_line(lwd = 1.5) + 
  labs(x = "Date", y = "Daily Average Depth (feet)") +
  theme(axis.title=element_text(size=18), # Text size 
        axis.text=element_text(size=18))

P33_1980_2000

ggsave(P33_1980_2000,
       filename = "Figures/P33_Time_Series_Missingness_1980_to_2000.png",
       scale = 2,
       width = 2000,
       height = 1000,
       units = 'px')

P33$Depth_Imputed = na.interp(P33$Depth)

P33_1980_2000_imputed = P33 %>% 
  filter(date >= "1980-01-01" & date <= "2000-01-01") %>% 
  ggplot(aes(x = date)) +
  geom_line(aes(y = Depth_Imputed), color = 'orange', lwd = 1.5) +
  geom_line(aes(y = Depth), color = 'black', lwd = 1.5) +
  labs(x = "Date", y = "Daily Average Depth (feet)") +
  theme(axis.title=element_text(size=18), # Text size 
        axis.text=element_text(size=18))

P33_1980_2000_imputed

ggsave(P33_1980_2000_imputed,
       filename = "Figures/P33_Time_Series_Missingness_1980_to_2000_Imputed.png",
       scale = 2,
       width = 2000,
       height = 1000,
       units = 'px')

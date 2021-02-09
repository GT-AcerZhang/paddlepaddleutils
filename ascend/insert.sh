awk 'BEGIN {
         RS = "\n" ;
         FS = "" ;
         d = 0 ;
     }

     {
         for (i=1; i<=NF; i++)
             if ($i == "{") {
                 d++ ;
                 if (d == 1) {
                     printf "{\n"
                     printf "\t[__FUNCTION__, __LINE__]"
                 }
             } else
             if ($i == "}") {
                 d-- ;
                 if (d == 0) printf "}"
             } else
             #if (d == 0)
             printf "%s", $i ;
         #if (d == 0) printf "\n"
         printf "\n"
     }' ./test_init.cc

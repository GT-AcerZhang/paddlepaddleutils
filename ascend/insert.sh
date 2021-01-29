awk 'BEGIN {
         RS = "\n" ;
         FS = "" ;
         d = 0 ;
     }

     {
         for (i=1; i<=NF; i++)
             if ($i == "{") {
                 d++ ;
                 if (d == 1) printf "{\n"
             } else
             if ($i == "}") {
                 d-- ;
                 if (d == 0) printf "}"
             } else
             if (d == 0)
                 print("__line_no__")
             printf "%s", $i ;
         if (d == 0) printf "\n"
     }' ./test_init.cc

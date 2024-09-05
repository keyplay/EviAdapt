for file in *_train.pt; do
  mv "$file" "$(echo $file | sed 's/\(.*\)_train\.pt/train_\1.pt/')"
done

for file in *_test.pt; do
  mv "$file" "$(echo $file | sed 's/\(.*\)_test\.pt/test_\1.pt/')"
done


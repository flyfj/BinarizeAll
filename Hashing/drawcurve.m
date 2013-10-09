

a = load('pr_curve0.txt');
b = load('pr_curve.txt');

xlabel('Recall')
ylabel('Precision')
hold on
plot(a(:,1), a(:,2), 'r-')
hold on
plot(b(:,1), b(:,2), 'b-')
hold on
legend('lsh', 'owh')
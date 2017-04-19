# multi-task-learning
multi task learning

이 글은 다음 글을 번역하고 몇 가지 주석을 단 것입니다.  
[http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/main.html](http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/main.html)

## 참고

## 개요

인공지능에서 중요한건 연속적으로 배우는것, 즉 old task를 까먹지 않고 new task를 배우는 것이다.  
역사적으로 신경망은 이걸 잘 못했다. McCloskey and Cohen (1989)<sup class="footnote-ref" id="fnref-1">[1](#fn-1)</sup>가 처음으로 보였는데,  
어떤 신경망에 1을 더하는 것을 가르치고, 그 다음 2를 더하는 것을 가르치면 1을 더하지 못한다.  
이렇게 신경망이 new task를 배우면서 빠르게 overwrite 해버리고 old task를 위한 parameter들을 잊어버리는 걸 catastrophic forgetting라고 부른다.

기존의 두 paper<sup class="footnote-ref" id="fnref-2">[2](#fn-2)</sup> 는 모든 task의 data가 훈련과정에서 available하도록 하여 좋은 결과를 냈다.  
하지만 task들이 sequential 하게 주어지면 (동시가 아니라, 순서대로 훈련해야되면) multitask learning은 episodic memory를 써서 training data를 record & replay 하는 방법밖에 없다.  
이러한 방법을 **system-level consolidation**이라고 부르는데, task의 수가 증가할수록 저장되는 memory의 수가 증가하여 전체 memory system이 커지게 된다.

이렇게 연속적인 학습을 거-대한 메모리 공간으로 해결하는 건 잘못되었다는 느낌이 든다. 우린 걸으면서 말하는 방법을 배울 때 어떻게 걷는 법을 배웠는지 기억하지 않았기 때문이다.  
그럼 포유류의 뇌는 어떻게 이걸 해낼까?

## 뉴런

![](http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/ewc_neuron_dendrites_spines.png)

*   dendrite = 수상돌기
*   dendritic spine: 뉴런의 dendrite에서 나온 돌기로, synapse 공간에서 single axon으로부터 input을 받는다. (그림의 빨간 원 참조)
*   Yang, Pan and Gan (2009)<sup class="footnote-ref" id="fnref-3">[3](#fn-3)</sup>는 postsynaptic dendritic spines의 형성과 제거로 학습이 이루어진다고 말했다. (오른쪽 그림 참조))
*   쥐들이 new task들에 대해 이동전략을 배우면서, 최적화될수록 spine 형성이 크게 증가하는 걸 관찰했다. 운동때문이라고 할 수 있기 때문에 따로 운동하는 비교군을 둬서 여기서는 spine 형성이 없음을 보였다.
*   또 새로 생긴 많은 spine 중 대부분은 제거되지만 일부는 남는다는 것을 관찰했다.
*   2015년 연구<sup class="footnote-ref" id="fnref-4">[4](#fn-4)</sup>에 따르면 특정 spine이 지워질 때 상응하는 skill을 잃는다는 것을 보였다.

## Overcoming catastrophic forgetting in neural networks<sup class="footnote-ref" id="fnref-0">[5](#fn-0)</sup>

*   "이 실험들에 의하면 neocortex (뇌의 부분)에서 연속적인 학습은 task-specific synapse의 구성으로 수행된다. 일부 synapse를 덜 plastic 하게 만들어서 지식을 기록한다."
*   plastic -> plasticity = 塑性 = (물리) 소성 = (물리) 가소성 = 힘을 받아 형태가 바뀐 뒤 되돌아가지 않는 성질
*   Q. 신경망에서 개별 뉴런의 중요도에 따라 다른 plasticity를 주는 비슷한 방법을 써서 catastrophic forgetting을 해결할 수 있을까?
*   A. 그렇다.

    ## 직관

(주의: 읽을 때 configuration과 parameter는 대충 같은 의미로 쓰이지만, parameters라고 하면 여러 \(\theta\\)를 의미하는게 아니라 한 \(\theta\\) 벡터의 여러 수를 의미하는 것이다. parameter 1개 = 수 1개, configuration 1개 = a vector of parameters = 1개의 \(\theta\\))

우리에게 순서대로 학습하고 싶은 task A와 B가 있다고 해보자.  
신경망이 task를 배운다는 건 신경망의 weight와 bias들을 조절한다는 것이다. 기존 work들은 큰 network에서 많은 \(\theta\)의 configuration들이 비슷한 성능을 가진다는것을 보였다. (DLB책에서 network의 width가 N이면 최소 N!개의 optimal point가 있다고 했던걸 떠올리면 된다.) 이건 결국 network가 overparameterized 되었다는 뜻이므로, B를 위한 \(\theta\) 중 A를 위한 \(\theta\)에 가까운 점이 존재한다는 뜻이다.

![](http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/ewc_parameter_space.png)

그림에서 \(\theta_A^*\)는 A에 최적화된 configuration이다. 하지만 그 가까운 주변 (회색 영역) 얘들도 task A를 꽤 잘 수행할 것이다. (원이 아니라 타원인 이유는 어떤 weights/biases들은 다른 얘들에 비해 더 많이 변화해야 같은 error를 증가시키기 때문이다.) 만약 이 상황에서 task A를 기억한다는 목적 없이 이어서 task B를 학습한다면 (즉 B의 gradient를 따라간다) 우리의 configuration는 파란 화살표를 따라갈 것이다.

하지만 우리는 A를 기억하고 싶다. 만약 단순히 모든 parameter들을 rigid하게 (\(\theta_A^*\)로부터 잘 안 바뀌게) 만든다면 우린 초록 화살표를 따라갈 것이고, task A와 B 모두 잘 못 해낼 것이다. 더 좋은 방법은 parameter들을 중요도에 따라 더 rigid하거나 덜 rigid하게 만들어서, 빨간 화살표를 따라가 task A와 B 모두 잘 수행하는 configuration을 찾는 것이다. 논문저자는 이걸 Elastic Weight Consolidation (EWC) 라고 부른다. 이 이름은 synaptic consolidatoin에서 따왔다. elastic이란 이름은 parameter들을 기존 solution에 묶는 constraint가 quadratic하기 때문이다.

## 수학

$p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}$\log p(\theta | \mathcal{D}) = \log p(\mathcal{D} | \theta) + \log p(\theta) - \log p(\mathcal{D})$

여기서 데이터 D는 2개의 independent parts로 나눌 수 있다. task A의 데이터 \(\mathcal{D}_A\)와 task B의 데이터 \(\mathcal{D}_B\)이다. 물론 여러 task도 가능하지만 여기선 2개만 보도록 하자.

$ \log p(\theta | \mathcal{D}) = \log \big[p(\mathcal{D}_A | \theta) p(\mathcal{D}_B | \theta)\big] + \log p(\theta) - \log \big[ p(\mathcal{D}_A) p(\mathcal{D}_B) \big]\nonumber\\ = \log p(\mathcal{D}_A | \theta) + \log p(\mathcal{D}_B | \theta) + \log p(\theta) - \log p(\mathcal{D}_A) - \log p(\mathcal{D}_B)$

A에 대해 묶어주면

$\log p(\theta | \mathcal{D}) = \log p(\mathcal{D}_A | \theta) + \log p(\theta | \mathcal{D}_A) - \log p(\mathcal{D}_B)$

우변의 \(p(\theta | \mathcal{D}_A)\) 은 task A를 풀 때 배운 정보를 담고 있다. 즉 이 probability가 task A를 풀 때 어떤 parameter들이 중요한지를 말해줄 수 있다.

다음 부분은 좀 어려우니 차근차근 살펴보자.

    "The true posterior probability is intractable, so, following the work on the Laplace approximation by Mackay (19), we approximate the posterior as a Gaussian distribution with mean given by the parameters $\theta_A^*$ and a diagonal precision given by the diagonal of the Fisher information matrix $F$."

*   true posterior probability는 \(p(\theta | \mathcal{D}_A)\)을 의미하고, 여기에 쓰일

$p(\mathcal{D}_A) = \int p(\mathcal{D}_A | \theta ') p(\theta ') d \theta '$

에서 적분의 closed form이 없으므로 계산이 불가능하다는 뜻이다. 수치근사를 해야 하는데 parameter의 수에 대해 지수적으로 계산 시간이 늘어난다. 그러므로 deep neural network에서는 수치근사조차 불가능하다.

*   Mackay's work on Laplace approximation: posterior probability를 수치근사하는 대신, multivariate normal distribution (mean = \(\theta_A^*\))으로 모델링한다. variance 모델링으로는 각 parameter마다 precision(=1/variance)을 정할 건데, precision 계산을 위해 Fisher information matrix \(F\)를 이용한다. Fisher information matrix는 수치근사보다 훨씬 계산이 쉽다. Fisher Information은 "랜덤변수 X가 안 알려진 parameter \(\theta\)에 대해 가진 정보량 (X가 \(\theta\)에 의존)"이다. 우리 상황에선 \(D_A\)의 각 sample이 \(\theta\)에 대해 가지는 정보량을 측정하는데 관심이 있다.

결과적으로 task A 뒤에 task B를 학습할 새 loss function을 정의해보자. \mathcal{L}_B(\theta)은 task B만 고려한 loss이다. 만약 parameter들을 i로 인덱싱하고 A와 B보다 얼마나 중요한지 나타내는 scalar \(\lambda\)를 골랐으면 EWC의 loss는 다음과 같다.

$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum\limits_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{A,i}^*)^2$

EWC는 parameter수와 training example 수에 linear한 수행시간을 가진다.

## 실험과 결과

### Random Binary Pattern to Binary

이전에 봤던 binary pattern을 기억하는지 측정한다.  
![](http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/ewc_num_patterns.png)  
EWC와 GD 둘 다 잘했지만 EWC가 더 많은 개수의 패턴을 기억했다.

TODO: 그래프 설명

### (modified) MNIST

*   Task A: 원래 MNIST 데이터의 label 0~9를 random permutation 한 task
*   Task B: 같음.
*   Task C: 같음.

A, B, C 순서대로 훈련하면서 A, B, C에 대해 측정

TODO: 정확히 어떻게 했다는 것인가?

![](http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/ewc_mnist.png)  
SGD는 catastrophic forgetting을 보인다. 또 EWC를 SGD+Dropout과 비교해보니 다음과 같았다.  
task의 수가 증가할수록 SGD+dropout의 성능은 계속 떨어졌다.

![](http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/ewc_many_tasks.png)

### Atari 2600

이전에 DeepMind에서 Deep Q Network (DQN)이 다양한 Atari 2600 게임들에서 super-human 성능을 보여줬다. EWC이 reinforcement learning에서 어떨지를 보기 위해서 DQN이 EWC를 쓰도록 수정했다. 여기에 추가 수정이 필요했는데, 포유류의 continual learning에서는 higher-level 시스템이 지금 agent가 어떤 task를 배우고 있는지를 정해주는 게 필요했는데, DQN에서는 그런 결정을 할 방법이 없었다. 그래서 forget-me-not(FMN) process에 기반한 online clustering algorithm을 추가해서, DQN agent가 각 inferred task에 대해 별도의 short-term memory buffer를 가지도록 하였다.

결과적으로 DQN agent들이 2개의 timescale 공간에서 학습하게 되었는데, short-term에선 SGD같은 optimizer를 써서 experience replay mechanism으로부터 배우고, long term에서는 여러 task에 걸쳐 배운 것을 EWC로 consolidate하였다. 10개 게임이 랜덤하게 선택되었고 각 게임은 agent에 일정시간동안 제공되었다.  
![](http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/ewc_training_schedule.png)

아래는 3개의 DQN agent들을 비교한 것이다.

*   blue: no EWC
*   red: EWC + forget-me-not task identifier (현재 어떤 task를 배우는지 추측함)
*   tan: EWC + true task label (현재 어떤 task를 배우는지 알려줌)
*   task 1개에 대한 사람 수준의 성능을 1로 normalize

결과:

*   EWC는 10개 게임에 대해 사람에 가까운 성능을 냈다.
*   non-EWC는 1개 task를 초과해서 배우지 못했다.
*   현재 배우는 task를 추측하거나 알려주거나는 별 차이가 없었다. (이건 EWC가 잘한게 아니라 FMN이 잘한 것이다)

![](http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/ewc_human_scores.png)

그런데 위 그래프를 보면 EWC가 10개 task에 대해 human-level에 다다르지는 못한다. (human-level=1점*10=10점)  
아마 Fisher information matrix가 parameter들의 중요도를 측정한흔데 좋은 방법이 아닌 것일까? 논문저자들은 실험적으로 이 가설을 체크하기 위해, 한 게임에서 훈련된 agent의 weights를 가지고 이리저리 관찰해보았다. 그 결과 어느 게임이냐에 상관 없이 다음 패턴을 발견했다: 만약 weights들이 uniform random perturbation에 영향을 받으면 pertubation의 크기가 커질수록 agent의 성능은 감소했으나, 만약 weights들이 **inverse of diagonal of the Fisher information**으로 모델링 된 perturbation에 영향을 받으면 perturbation이 꽤 커져도 같은 점수를 유지했다. 즉 Fisher information은 진짜 중요한 parameter를 중요하다고 표현하는데 좋은 방법이다.

그런데, 논문저자들은 이번엔 null space에서 perturbation을 줘 봤다. 아무런 영향이 없어야 하지만 위에서 inverse Fisher space를 썼을 때와 비슷한 결과를 관찰했다. 즉 Fisher information matrix를 썼더니 일부 중요한 parameter를 중요하지 않다고 labeling한 것이다. "현재 구현의 주 문제점은 parameter의 불확실성을 과소평가하기 때문인 것 같다"

![](http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/ewc_perturbation.png)

## 토의

*   EWC는 멋진 Bayesian 해석을 할 수 있다: "배워야 할 new task가 있을 때, 이전 task들의 데이터들이 주어졌을 때의 parameters의 posterior distribution을 prior로 하여 network parameters를 완화시킨다."

*   Network Overlap:  
    글의 처음부분에서 우린 신경망의 over-parameterization이 EWC가 좋은 성능을 얻게 해 준다고 말했었다. 다음과 같은 질문을 해 볼 수 있다: 네트워크를 section들로 쪼개서 각 section을 각 task에 활용하면 더 좋을까? 아니면 네트워크가 representation을 공유해서 capacity를 효율적으로 잘 사용하고 있는 것일까? 답을 위해 논문저자들은 tasks 쌍들간의 (pair들 간의) overlap을 측정했다. (Fisher overlap<sup class="footnote-ref" id="fnref-6">[6](#fn-6)</sup> 개념 참고) 매우 가까운 task들에 대해서는 Fisher overlap이 매우 높았고, 비슷하지 않은 task들에 대해서도 Fisher overlap은 0보다는 한참 컸다. network의 depth가 커질수록 Fisher overlap도 커졌다.

![](http://rylanschaeffer.github.io/content/research/overcoming_catastrophic_forgetting/ewc_fisher_overlap.png)

*   Synaptic Plasticity:  
    어떻게 EWC가 계산 모델에게 synaptic plasticity를 알려줄 수 있었을까? cascade 이론은 plasticity와 stability를 모델링 하기 위해 synaptic state 모델을 만든다. EWC는 시간에 따라 이전 정보를 잊는 게 불가능하긴 하지만, EWC와 cascade 모두 memory stability는 synapse들을 less plastic하게 만들어서 얻을 수 있다는 데 동의한다. 더 최근 논문[^6]에서는 synapse들이 weight를 저장할 뿐만 아니라 현재 weight에 대한 uncertainty도 저장한다고 제안한다. EWC는 이 아이디어의 확장이다: 각 synapse는 3개를 저장한다: weight, mean, variance.

TODO: EWC와 cascade의 차이는?

## 결론

*   잊지 않고 여러 task를 순서대로 학습하는 건 지능 (인공지능이던 아니던) 에 필요하다.
*   연구들에 의하면 포유류 뇌의 synaptic consolidation이 continual learning을 해낸다.
*   EWC은 중요한 parameter들을 less plastic하게 만들어서 synaptic consolidation을 흉내낸다.
*   EWC가 적용된 neural network는 여러 domain에서 SGD & uniform parameter stability보다 좋은 성능을 보였다.
*   EWC는 weight consolidation이 continual learning의 기초가 된다는 걸 보였다.

<div class="footnotes">

* * *

1.  [http://www.sciencedirect.com/science/article/pii/S0079742108605368](http://www.sciencedirect.com/science/article/pii/S0079742108605368)[↩](#fnref-1)

2.  [https://arxiv.org/pdf/1511.06342.pdf](https://arxiv.org/pdf/1511.06342.pdf)[↩](#fnref-2)

3.  [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4724802/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4724802/)[↩](#fnref-3)

4.  [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4634641/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4634641/)[↩](#fnref-4)

5.  [http://www.pnas.org/content/early/2017/03/13/1611835114.full.pdf?with-ds=yes](http://www.pnas.org/content/early/2017/03/13/1611835114.full.pdf?with-ds=yes)[↩](#fnref-0)

6.  [http://www.pnas.org/content/suppl/2017/03/14/1611835114.DCSupplemental/pnas.201611835SI.pdf](http://www.pnas.org/content/suppl/2017/03/14/1611835114.DCSupplemental/pnas.201611835SI.pdf)[↩](#fnref-6)

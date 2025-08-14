import { useId, type FC, type InputHTMLAttributes } from "react";

type State = InputHTMLAttributes<HTMLInputElement> & {
	label: string;
	parentClassName?: string;
};

const InputFloating: FC<State> = (props) => {
	const { className = "", parentClassName = "", label, ...rest } = props;

	const inputId = useId();
	return (
		<div className={`relative ${parentClassName}`}>
			<input
				type="email"
				id={inputId}
				className={`peer block w-full rounded-lg border-gray-200 p-4 not-placeholder-shown:pt-6 not-placeholder-shown:pb-2 placeholder:text-transparent autofill:pt-6 autofill:pb-2 focus:border-blue-500 focus:pt-6 focus:pb-2 focus:ring-blue-500 disabled:pointer-events-none disabled:opacity-50 sm:text-sm ${className}`}
				placeholder={label}
				{...rest}
			/>
			<label
				htmlFor={inputId}
				className="pointer-events-none absolute start-0 top-0 h-full origin-[0_0] truncate border border-transparent p-4 transition duration-100 ease-in-out peer-not-placeholder-shown:translate-x-0.5 peer-not-placeholder-shown:-translate-y-1.5 peer-not-placeholder-shown:scale-90 peer-not-placeholder-shown:text-gray-500 peer-focus:translate-x-0.5 peer-focus:-translate-y-1.5 peer-focus:scale-90 peer-focus:text-gray-500 peer-disabled:pointer-events-none peer-disabled:opacity-50 sm:text-sm"
			>
				{label}
			</label>
		</div>
	);
};

export default InputFloating;
